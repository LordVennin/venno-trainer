"""Tagger: a tiny random model in the WD v3 file layout, the prediction
format, thumbnails and the stdin/stdout service protocol."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import timm  # noqa: F401
except ImportError:  # pragma: no cover
    timm = None


def make_wd_model(models_dir):
    import torch
    from safetensors.torch import save_file
    from venno.tagger import models as M

    root = M.model_dir(models_dir, "wd-vit-v3")
    os.makedirs(root, exist_ok=True)
    cfg = {
        "architecture": "vit_tiny_patch16_224",
        "num_classes": 4,
        "model_args": {"img_size": 64, "class_token": False, "global_pool": "avg", "fc_norm": False},
        "pretrained_cfg": {"input_size": [3, 64, 64]},
    }
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(0)
    model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=4, **cfg["model_args"])
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(root, "model.safetensors"))
    with open(os.path.join(root, "selected_tags.csv"), "w") as f:
        f.write("tag_id,name,category,count\n1,general,9,1\n2,long_hair,0,1\n3,^_^,0,1\n4,some_character,4,1\n")
    return root


class PickingTaggerTests(unittest.TestCase):
    """run() with a tagger that picks its own tags, as Hydra 3.5 does."""

    def test_picked_tags_are_marked_and_aliased(self):
        import torch
        from venno.tagger import models as M

        class Fake:
            tags = [("vulva", "general"), ("dragon", "species"), ("wings", "general"), ("safe", "rating")]
            aliases = {"vulva": "pussy"}

            def preprocess(self, path):
                return torch.zeros(1)

            def predict(self, batch):
                return torch.tensor([[0.05, 0.9, 0.5, 0.99]] * len(batch))

            def classify(self, row, opts):
                assert opts["confidence"] == 0.8
                # implications may raise a tag's probability or add a tag below the floor
                return {"dragon": 0.9, "vulva": 0.95}

        out = M.run(Fake(), ["a.png"], floor=0.3, options={"confidence": 0.8, "aliases": True})[0]["preds"]
        self.assertEqual(out, [["safe", 0.99, "rating", False], ["pussy", 0.95, "general", True],
                               ["dragon", 0.9, "species", True], ["wings", 0.5, "general", False]])
        out = M.run(Fake(), ["a.png"], floor=0.3, options={"confidence": 0.8})[0]["preds"]
        self.assertEqual(out[1][0], "vulva")


@unittest.skipIf(timm is None, "timm not installed")
class TaggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="venno_tagger_")
        cls.models = os.path.join(cls.tmp, "models")
        make_wd_model(cls.models)
        cls.images = []
        for i, (mode, size) in enumerate((("RGB", (80, 50)), ("RGBA", (40, 90)), ("P", (64, 64)))):
            p = os.path.join(cls.tmp, f"img{i}.png")
            Image.new(mode, size).save(p)
            cls.images.append(p)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_predictions(self):
        from venno.tagger import models as M

        self.assertTrue(M.is_downloaded(self.models, "wd-vit-v3"))
        tagger = M.load("wd-vit-v3", self.models, "cpu", status=lambda s: None)
        out = M.run(tagger, self.images + [os.path.join(self.tmp, "missing.png")], floor=0.0, batch_size=2)
        self.assertEqual(len(out), 4)
        self.assertIn("error", out[3])
        for r in out[:3]:
            names = [p[0] for p in r["preds"]]
            self.assertEqual(sorted(names), ["^_^", "general", "long_hair", "some_character"])
            cats = {p[0]: p[2] for p in r["preds"]}
            self.assertEqual(cats, {"general": "rating", "long_hair": "general", "^_^": "general", "some_character": "character"})
            probs = [p[1] for p in r["preds"]]
            self.assertEqual(probs, sorted(probs, reverse=True))
        # white padding, BGR order and [-1, 1] range
        x = tagger.preprocess(self.images[0])
        self.assertEqual(tuple(x.shape), (3, 64, 64))
        self.assertAlmostEqual(float(x[:, 0, 0].min()), 1.0, places=5)

    def test_service(self):
        thumb = os.path.join(self.tmp, "thumbs", "a.webp")
        reqs = [
            {"id": 1, "cmd": "thumbs", "items": [{"src": self.images[1], "dst": thumb, "size": 32}]},
            {"id": 2, "cmd": "tag", "model": "wd-vit-v3", "device": "cpu", "paths": self.images[:1], "models_dir": self.models, "floor": 0.0},
            {"id": 3, "cmd": "status"},
            {"id": 4, "cmd": "unload"},
            {"id": 5, "cmd": "nope"},
        ]
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE))
        proc = subprocess.run([sys.executable, "-m", "venno.tagger"], input="\n".join(json.dumps(r) for r in reqs) + "\n",
                              capture_output=True, text=True, env=env, timeout=300)
        lines = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(lines[0], {"event": "ready"})
        by_id = {m["id"]: m for m in lines if "id" in m}
        self.assertTrue(by_id[1]["ok"])
        with Image.open(thumb) as im:
            self.assertLessEqual(max(im.size), 32)
        self.assertTrue(by_id[2]["ok"])
        self.assertEqual(by_id[2]["device"], "cpu")
        self.assertEqual(len(by_id[2]["results"][0]["preds"]), 4)
        self.assertTrue(by_id[4]["ok"])
        self.assertFalse(by_id[5]["ok"])
        self.assertTrue(any(m.get("event") == "status" for m in lines))


if __name__ == "__main__":
    unittest.main()
