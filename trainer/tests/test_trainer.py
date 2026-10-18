"""End-to-end CPU tests on tiny random models in the real checkpoint layouts.

    cd trainer && python -m unittest discover -s tests -v
"""

import json
import os
import random
import shutil
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tests import fixtures as F  # noqa: E402
from venno import data as D  # noqa: E402
from venno import train as T  # noqa: E402

os.environ["VENNO_DEVICE"] = "cpu"
torch.set_num_threads(4)


def run_config(job, model_paths, out, cache):
    return {"job": job, "model_paths": model_paths, "output_dir": out, "cache_dir": cache, "models_dir": cache}


def train(run):
    T._stop_requested = False
    T.Trainer(run).train()


class DataTests(unittest.TestCase):
    def test_caption_processing_order(self):
        f = D.Folder(path="x", caption_prefix="sks,", keep_tokens=1, shuffle=True, tag_dropout=0.5)
        item = D.Item(image="a.png", caption="a, b, c, d, e", folder=f)
        texts = {D.caption_variant(item, k, 8, 42) for k in range(8)}
        self.assertGreater(len(texts), 1)
        for t in texts:
            self.assertTrue(t.startswith("sks"), t)  # keep_tokens keeps the prefix tag first
        # deterministic per (seed, image, k)
        self.assertEqual(D.caption_variant(item, 3, 8, 42), D.caption_variant(item, 3, 8, 42))
        plain = D.Item(image="b.png", caption="a, b", folder=D.Folder(path="x", caption_prefix="sks,"))
        self.assertEqual(D.caption_variant(plain, 0, 8, 42), "sks, a, b")

    def test_dropout_every_n_epochs(self):
        f = D.Folder(path="x", dropout_every_n_epochs=3)
        rng = random.Random(0)
        self.assertTrue(D.is_dropped(f, 3, rng))
        self.assertFalse(D.is_dropped(f, 4, rng))

    def test_buckets_and_epoch_batches(self):
        a, b, r = D.Folder(path="a", repeats=3), D.Folder(path="b", repeats=1), D.Folder(path="r", is_reg=True)
        items = [D.Item(f"{x}{i}", "", fo, size=(640, 480)) for x, fo, n in (("a", a, 4), ("b", b, 2), ("r", r, 3)) for i in range(n)]
        D.assign_buckets(items, [512, 768], 64, 256, 1024, True, False)
        self.assertEqual(items[0].buckets[0][0] % 64, 0)
        self.assertGreater(items[0].buckets[0][0], items[0].buckets[0][1])  # landscape stays landscape
        batches = D.epoch_batches(items, [2, 1], 2, random.Random(0))
        samples = [s for bt in batches for s in bt]
        train = 4 * 3 + 2 * 1
        self.assertEqual(len(samples), 2 * (train + train))  # reg matched to train count, per resolution
        self.assertEqual(sum(1 for s in samples if s.item.folder is a), 2 * 12)
        for bt in batches:
            self.assertEqual(len({s.bucket for s in bt}), 1)

    def test_no_upscale_small_image(self):
        self.assertEqual(D.choose_bucket((300, 200), 1024, 64, D.make_buckets(1024, 64, 256, 2048), True, True), (256, 192))


class AnimaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="venno_anima_")
        cls.paths = F.make_anima(os.path.join(cls.tmp, "models"))
        F.make_images(os.path.join(cls.tmp, "data", "char"), 3, caption=lambda i: f"sks, 1girl, tag{i}, smile")
        F.make_images(os.path.join(cls.tmp, "data", "style"), 2, seed=1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def job(self, **training):
        job = F.base_job("anima", [
            {"path": os.path.join(self.tmp, "data", "char"), "repeats": 2, "caption_dropout": 0.3, "flip": True},
            {"path": os.path.join(self.tmp, "data", "style"), "tag_dropout": 0.3, "shuffle": True, "keep_tokens": 1},
        ], **training)
        job["dataset"].update(resolutions="64", batch_sizes="2", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["training"].update(optimizer="AdamW", lr="1e-3", output_name="anima_test", workers=1)
        job["sampling"].update(prompts=[{"text": "sks, 1girl", "width": 64, "height": 64, "steps": 2, "cfg": 2.0, "seed": 1}])
        return job

    def test_train_save_resume(self):
        out = os.path.join(self.tmp, "out1")
        cache = os.path.join(self.tmp, "cache")
        job = self.job(duration_unit="steps", max_steps=4, save_every_steps=2, cache_te=True, te_variants=3)
        job["sampling"].update(every_steps=2)
        train(run_config(job, self.paths, out, cache))

        final = load_file(os.path.join(out, "anima_test.safetensors"))
        keys = sorted(final)
        self.assertIn("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight", keys)
        self.assertIn("lora_unet_blocks_0_adaln_modulation_mlp_1.alpha", keys)
        self.assertIn("lora_unet_blocks_0_mlp_layer2.lora_up.weight", keys)
        self.assertTrue(all(k.startswith("lora_unet_blocks_") for k in keys))
        self.assertEqual(final["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"].shape[0], 16)
        self.assertGreater(final["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"].abs().sum(), 0)  # it trained
        self.assertTrue(os.path.exists(os.path.join(out, "anima_test-step00000002.safetensors")))
        samples = os.listdir(os.path.join(out, "samples"))
        self.assertTrue(any("_000002_" in s for s in samples) and any("_final_" in s for s in samples), samples)

        # caches: 5 images x (1 bucket); text: dropout "", 3 char captions, 2 style x 3 variants, prompt, negative
        n_text = sum(len(f) for _, _, f in os.walk(os.path.join(cache, "anima", "text")))
        self.assertGreaterEqual(n_text, 1 + 3 + 2 * 2 + 1)
        with safe_open(next(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(cache, "anima", "latents")) for f in fs), "pt") as fh:
            self.assertEqual(fh.get_tensor("latent").shape[0], 16)

        # resume continues from the final state up to the new step count
        job2 = self.job(duration_unit="steps", max_steps=6, save_every_steps=100)
        job2["network"]["auto_resume"] = True
        job2["sampling"]["enabled"] = False
        out_lines = []
        orig = T.log
        T.log = lambda *a: (out_lines.append(" ".join(map(str, a))), orig(*a))
        try:
            train(run_config(job2, self.paths, out, cache))
        finally:
            T.log = orig
        self.assertTrue(any("Resumed from" in l and "step 4" in l for l in out_lines), out_lines)
        self.assertTrue(any("Latents: all cached" in l for l in out_lines), out_lines)
        resumed = load_file(os.path.join(out, "anima_test.safetensors"))
        self.assertFalse(torch.equal(resumed["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"],
                                     final["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"]))

    def test_live_text_epochs_and_init_weights(self):
        out = os.path.join(self.tmp, "out2")
        job = self.job(duration_unit="epochs", max_epochs=2, save_every_epochs=1, cache_te=False)
        job["sampling"]["enabled"] = False
        job["network"].update(dim=4, alpha=2, ignore_layers="adaln_modulation")
        train(run_config(job, self.paths, out, os.path.join(self.tmp, "cache2")))
        sd = load_file(os.path.join(out, "anima_test.safetensors"))
        self.assertFalse(any("adaln_modulation" in k for k in sd))
        self.assertTrue(os.path.exists(os.path.join(out, "anima_test-000001.safetensors")))

        job["network"]["init_weights"] = os.path.join(out, "anima_test.safetensors")
        job["training"].update(max_epochs=1, lr="0")
        job["training"]["output_name"] = "anima_init"
        train(run_config(job, self.paths, out, os.path.join(self.tmp, "cache2")))
        again = load_file(os.path.join(out, "anima_init.safetensors"))
        for k in sd:
            self.assertTrue(torch.allclose(sd[k].float(), again[k].float()), k)  # lr 0 keeps the loaded weights


class MossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="venno_moss_")
        try:
            tok = F.fetch_moss_tokenizer(os.path.join(cls.tmp, "hub"))
        except Exception as e:  # offline
            raise unittest.SkipTest(f"MOSS tokenizer not downloadable: {e}")
        cls.paths = F.make_moss(os.path.join(cls.tmp, "moss"), tok)
        F.make_images(os.path.join(cls.tmp, "data", "a"), 3, caption=lambda i: f"A photo of a red object number {i}.")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_train_keys_and_lora_roundtrip(self):
        out = os.path.join(self.tmp, "out")
        job = F.base_job("moss_image", [{"path": os.path.join(self.tmp, "data", "a"), "repeats": 2, "loss_weight": 2.0}],
                         duration_unit="steps", max_steps=3, save_every_steps=10, optimizer="AdamW", lr="1e-3",
                         output_name="moss_test", workers=1)
        job["dataset"].update(resolutions="64, 96", batch_sizes="1", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["network"].update(dim=8, alpha=8, ignore_layers="layers.1.")
        job["sampling"].update(prompts=[{"text": "a red object", "width": 64, "height": 64, "steps": 3, "cfg": 3.0, "seed": 3}],
                               every_steps=100)
        run = run_config(job, self.paths, out, os.path.join(self.tmp, "cache"))
        train(run)
        sd = load_file(os.path.join(out, "moss_test.safetensors"))
        self.assertIn("diffusion_model.layers.0.proj.q.lora_A.weight", sd)
        self.assertIn("diffusion_model.layers.0.proj.v.lora_B.weight", sd)
        self.assertIn("diffusion_model.layers.2.proj.ffn_up_gate.lora_A.weight", sd)
        self.assertIn("diffusion_model.layers.0.proj.o.alpha", sd)
        self.assertFalse(any(".layers.1." in k for k in sd))
        self.assertFalse(any(".qkv." in k for k in sd))
        self.assertEqual(sd["diffusion_model.layers.0.proj.q.lora_B.weight"].shape, (128, 8))
        self.assertTrue(torch.equal(sd["diffusion_model.layers.0.proj.q.lora_A.weight"], sd["diffusion_model.layers.0.proj.k.lora_A.weight"]))
        self.assertTrue(any(n.endswith(".png") for n in os.listdir(os.path.join(out, "samples"))))

        # export -> import is lossless (q/k/v fused back)
        trainer = T.Trainer(run)
        internal = trainer.adapter.import_lora(sd)
        self.assertIn("layers.0.proj.qkv.lora_up.weight", internal)
        self.assertEqual(internal["layers.0.proj.qkv.lora_up.weight"].shape, (3 * 128, 8))
        back = trainer.adapter.export_lora(internal)
        self.assertEqual(sorted(back), sorted(sd))
        for k in sd:
            self.assertTrue(torch.equal(sd[k], back[k]), k)


if __name__ == "__main__":
    unittest.main()
