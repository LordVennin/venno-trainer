"""Tagger service for the UI: JSON requests on stdin, one per line, answered
on stdout as {"id": ..., "ok": true, ...} or {"id": ..., "ok": false,
"error": ...}. Lines without an id ({"event": "status", ...}) report model
download/load progress. The model stays loaded between requests until
"unload" (or the process exits), so tagging one image at a time is fast.

Requests:
  {"cmd": "tag", "model": id, "device": "cuda"|"cpu", "paths": [...],
   "models_dir": dir, "floor": 0.1, "batch_size": 4, "options": {...}}
  (options: Hydra 3.5's confidence, implications, aliases, exclude_categories
   and exclude_tags)
  {"cmd": "thumbs", "items": [{"src": path, "dst": path, "size": 320}]}
  {"cmd": "unload"} / {"cmd": "status"} / {"cmd": "models", "models_dir": dir}
Model work runs on one worker thread (in order); thumbnails on a pool."""

import json
import os
import queue
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

_out = sys.stdout
sys.stdout = sys.stderr  # stray prints from libraries must not break the protocol
_write_lock = threading.Lock()


def send(msg):
    line = json.dumps(msg, ensure_ascii=False)
    with _write_lock:
        _out.write(line + "\n")
        _out.flush()


def status(text):
    send({"event": "status", "text": text})


def make_thumb(src, dst, size=320):
    """WebP thumbnail with EXIF rotation; JPEGs decode at reduced size."""
    with Image.open(src) as im:
        if im.format == "JPEG":
            im.draft("RGB", (size * 2, size * 2))
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "transparency" in im.info or im.mode in ("LA", "PA") else "RGB")
        im.thumbnail((size, size), Image.BICUBIC, reducing_gap=2.0)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        im.save(tmp, "WEBP", quality=82, method=4)
        os.replace(tmp, dst)


class Service:
    def __init__(self):
        self.tagger = None
        self.loaded = None  # (model_id, device)
        self.jobs = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) // 2)))
        self.threads = []
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while True:
            msg = self.jobs.get()
            if msg is None:
                return
            self._answer(msg, self._model_cmd)

    def finish(self):
        """Answer everything already asked before exiting (stdin closed)."""
        self.jobs.put(None)
        self.worker.join()
        for t in self.threads:
            t.join()

    def _answer(self, msg, fn):
        try:
            send({"id": msg.get("id"), "ok": True, **fn(msg)})
        except Exception as e:
            traceback.print_exc()
            send({"id": msg.get("id"), "ok": False, "error": str(e) or type(e).__name__})

    def _unload(self):
        if self.tagger is None:
            return
        self.tagger = None
        self.loaded = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _model_cmd(self, msg):
        from . import models as M

        cmd = msg["cmd"]
        if cmd == "unload":
            self._unload()
            return {}
        if cmd == "tag":
            want = (msg["model"], msg.get("device") or "cpu")
            if want[1] == "cuda":
                import torch

                if not torch.cuda.is_available():
                    want = (want[0], "cpu")
            if self.loaded != want:
                self._unload()
                self.tagger = M.load(want[0], msg.get("models_dir") or "models", want[1], msg.get("hf_token"), status)
                self.loaded = want
                status(f"Ready: {M.MODELS[want[0]]['label']} on {want[1]}")
            results = M.run(self.tagger, list(msg.get("paths") or []), floor=float(msg.get("floor", 0.1)),
                            batch_size=max(1, int(msg.get("batch_size") or 4)), options=msg.get("options"))
            return {"results": results, "device": want[1]}
        raise ValueError(f"Unknown command: {cmd}")

    def _thumbs(self, msg):
        def one(item):
            try:
                if not os.path.exists(item["dst"]):
                    make_thumb(item["src"], item["dst"], int(item.get("size") or 320))
                return None
            except Exception as e:
                return f"{os.path.basename(item['src'])}: {e}"

        errors = [e for e in self.pool.map(one, msg.get("items") or []) if e]
        return {"errors": errors}

    def handle(self, msg):
        cmd = msg.get("cmd")
        if cmd in ("tag", "unload"):
            self.jobs.put(msg)
        elif cmd == "thumbs":
            t = threading.Thread(target=self._answer, args=(msg, self._thumbs), daemon=True)
            self.threads = [x for x in self.threads if x.is_alive()] + [t]
            t.start()
        elif cmd == "status":
            self._answer(msg, lambda m: {"loaded": list(self.loaded) if self.loaded else None})
        elif cmd == "models":
            from . import models as M

            def listing(m):
                d = m.get("models_dir") or "models"
                return {"models": [{"id": k, "label": v["label"], "kind": v["kind"], "gb": v["gb"],
                                    "downloaded": M.is_downloaded(d, k)} for k, v in M.MODELS.items()]}

            self._answer(msg, listing)
        else:
            self._answer(msg, lambda m: (_ for _ in ()).throw(ValueError(f"Unknown command: {cmd}")))


def main():
    svc = Service()
    send({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        svc.handle(msg)
    svc.finish()


if __name__ == "__main__":
    main()
