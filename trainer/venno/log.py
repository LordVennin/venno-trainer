"""Console output shared by the trainer: plain log lines plus a tqdm-style
progress line the UI parses ("123/1000 [01:23<10:00, 1.2it/s, avr_loss=0.09]")."""

import sys
import time

_progress_active = False


def log(*parts):
    global _progress_active
    if _progress_active:
        sys.stdout.write("\n")
        _progress_active = False
    print(*parts, flush=True)


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    def __init__(self, total, desc="steps", start=0):
        self.total = max(1, int(total))
        self.desc = desc
        self.start_step = start
        self.start_time = time.time()

    def update(self, step, loss=None, vram_gb=None):
        global _progress_active
        elapsed = time.time() - self.start_time
        done = step - self.start_step
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - step) / rate if rate > 0 else 0.0
        frac = min(1.0, step / self.total)
        bar = ("█" * int(frac * 20)).ljust(20)
        speed = f"{rate:.2f}it/s" if rate >= 1 else f"{(1 / rate) if rate else 0:.2f}s/it"
        # only training bars carry a loss (the UI charts lines with avr_loss)
        tail = f", avr_loss={loss:.4g}" if loss is not None else ""
        if vram_gb is not None:
            tail += f", vram={vram_gb:.1f}G"
        sys.stdout.write(
            f"\r{self.desc}: {int(frac * 100):3d}%|{bar}| {step}/{self.total} "
            f"[{_fmt_time(elapsed)}<{_fmt_time(eta)}, {speed}{tail}]"
        )
        sys.stdout.flush()
        _progress_active = True

    def close(self):
        global _progress_active
        if _progress_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _progress_active = False
