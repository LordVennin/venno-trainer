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

import numpy as np
import torch
from PIL import Image
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
        [(_, texts)] = D.caption_choices(item, 8, 42)
        self.assertEqual(len(texts), 8)
        self.assertGreater(len(set(texts)), 1)
        for t in texts:
            self.assertTrue(t.startswith("sks"), t)  # keep_tokens keeps the prefix tag first
        # deterministic per (seed, image, k)
        self.assertEqual(texts, D.caption_choices(item, 8, 42)[0][1])
        plain = D.Item(image="b.png", caption="a, b", folder=D.Folder(path="x", caption_prefix="sks,"))
        self.assertEqual(D.caption_choices(plain, 8, 42), [(1.0, ["sks, a, b"])])

    def test_tags_and_prose_lines(self):
        tmp = tempfile.mkdtemp()
        try:
            F.make_images(tmp, 2)
            with open(os.path.join(tmp, "img000.txt"), "w") as fh:
                fh.write("fox, solo, standing\nA red fox stands in the snow, looking back.\n")
            folder = D.Folder(path=tmp, caption_prefix="sks,", keep_tokens=1, shuffle=True, tag_dropout=0.3)
            items, _ = D.scan_folders([folder], ".txt")
            fox = items[0]
            self.assertEqual((fox.caption, fox.prose), ("fox, solo, standing", "A red fox stands in the snow, looking back."))
            self.assertEqual(fox.default_kind, "both")
            # tag ops touch only the tag line: the prose ends every variant intact
            [(_, texts)] = D.caption_choices(fox, 8, 0)
            for t in texts:
                self.assertTrue(t.startswith("sks"), t)
                self.assertTrue(t.endswith(". A red fox stands in the snow, looking back."), t)
            rng = random.Random(0)
            live = {D.caption_live(fox, 0, rng) for _ in range(30)}
            self.assertTrue(all(t.endswith("looking back.") for t in live))
            self.assertEqual(D.compose(fox, "prose"), "sks, A red fox stands in the snow, looking back.")
            # a model that reads only line 1 (Anima) trains on the tags without a mix
            first, _ = D.scan_folders([D.Folder(path=tmp)], ".txt", first_line_only=True)
            self.assertEqual(D.caption_choices(first[0], 8, 0), [(1.0, ["fox, solo, standing"])])
            # an image without a caption line trains on its one line, mix or not
            self.assertEqual(D.caption_kinds(items[1]), [("tags", 1.0)])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_caption_mix(self):
        self.assertEqual(D.parse_mix(""), ())
        self.assertEqual(D.parse_mix("25:25:50%"), (25.0, 25.0, 50.0))
        for bad in ("25:75", "a:b:c", "0:0:0", "-1:1:1"):
            with self.assertRaises(ValueError):
                D.parse_mix(bad)
        f = D.Folder(path="x", caption_mix=D.parse_mix("25:25:50"), shuffle=True)
        item = D.Item(image="a.png", caption="fox, solo", folder=f, prose="A fox.")
        choices = D.caption_choices(item, 8, 0)
        self.assertEqual([w for w, _ in choices], [25.0, 25.0, 50.0])
        # the tag-line forms share the variant budget: 8 in total, plus one prose
        self.assertEqual([len(t) for _, t in choices], [3, 1, 5])
        self.assertEqual(choices[1][1], ["A fox."])
        rng = random.Random(1)
        picks = [D.pick_caption(choices, rng) for _ in range(4000)]
        share = lambda pred: sum(map(pred, picks)) / len(picks)
        self.assertAlmostEqual(share(lambda t: t == "A fox."), 0.25, delta=0.03)
        self.assertAlmostEqual(share(lambda t: t.endswith("A fox.") and t != "A fox."), 0.5, delta=0.03)
        self.assertAlmostEqual(share(lambda t: "A fox" not in t), 0.25, delta=0.03)
        # a weight of 0 leaves that form out
        only = D.Folder(path="x", caption_mix=D.parse_mix("0:50:50"))
        self.assertEqual([k for k, _ in D.caption_kinds(D.Item("b.png", "fox", only, prose="A fox."))], ["prose", "both"])
        self.assertEqual(D.join_lines("fox, solo.", "A fox."), "fox, solo. A fox.")

    def test_dropout_every_n_epochs(self):
        # sd-scripts counts epochs from 1: N=3 drops captions on epochs 3, 6, ...
        f = D.Folder(path="x", dropout_every_n_epochs=3)
        rng = random.Random(0)
        dropped = [i + 1 for i in range(9) if D.is_dropped(f, i, rng)]
        self.assertEqual(dropped, [3, 6, 9])
        self.assertFalse(any(D.is_dropped(D.Folder(path="x"), i, rng) for i in range(9)))

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

    def test_uncaptioned_images_are_listed(self):
        tmp = tempfile.mkdtemp()
        try:
            F.make_images(tmp, 3)
            os.remove(os.path.join(tmp, "img001.txt"))
            items, missing = D.scan_folders([D.Folder(path=tmp)], ".txt")
            self.assertEqual(len(items), 3)
            self.assertEqual(missing, [os.path.join(tmp, "img001.png")])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
        self.assertFalse(any("adaln_modulation" in k for k in keys), "attention and MLP only by default")
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

        # EMA (on by default): the state carries the average of the 4 steps
        st = torch.load(os.path.join(out, "anima_test-final-state", "state.pt"), weights_only=False)
        self.assertEqual(st["ema"]["n"], 4)

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

    def test_ema_saves_the_average(self):
        out = os.path.join(self.tmp, "out_ema")
        job = self.job(duration_unit="steps", max_steps=5, save_every_steps=100, ema_decay=0.9)
        job["sampling"]["enabled"] = False
        trainer = T.Trainer(run_config(job, self.paths, out, os.path.join(self.tmp, "cache")))
        T._stop_requested = False
        trainer.train()
        saved = load_file(os.path.join(out, "anima_test.safetensors"))
        snap = lambda: {k: v.detach().clone() for k, v in trainer.adapter.export_lora(trainer.network.internal_state_dict()).items()}
        raw = snap()
        with trainer.ema.applied():
            avg = snap()
        k = "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"
        self.assertTrue(torch.allclose(saved[k].float(), avg[k].float().cpu(), atol=1e-2))
        self.assertFalse(torch.allclose(avg[k].float(), raw[k].float(), atol=1e-6), "the average differs from the last step")
        self.assertTrue(torch.equal(snap()[k], raw[k]), "training weights are restored after a save")
        # a float32 save on the CPU (no copy from the GPU) still writes the average
        job["training"]["save_precision"] = "fp32"
        trainer.t["save_precision"] = "fp32"
        trainer.save_lora(os.path.join(out, "fp32.safetensors"))
        self.assertTrue(torch.allclose(load_file(os.path.join(out, "fp32.safetensors"))[k], avg[k].float()))

    def test_live_text_epochs_and_init_weights(self):
        out = os.path.join(self.tmp, "out2")
        job = self.job(duration_unit="epochs", max_epochs=2, save_every_epochs=1, cache_te=False)
        job["sampling"]["enabled"] = False
        job["network"].update(dim=4, alpha=2, ignore_layers="")
        train(run_config(job, self.paths, out, os.path.join(self.tmp, "cache2")))
        sd = load_file(os.path.join(out, "anima_test.safetensors"))
        self.assertIn("lora_unet_blocks_0_adaln_modulation_mlp_1.alpha", sd, "a cleared field trains adaLN too")
        self.assertTrue(os.path.exists(os.path.join(out, "anima_test-000001.safetensors")))

        job["network"]["init_weights"] = os.path.join(out, "anima_test.safetensors")
        job["training"].update(max_epochs=1, lr="0", ema_decay=0)
        job["training"]["output_name"] = "anima_init"
        train(run_config(job, self.paths, out, os.path.join(self.tmp, "cache2")))
        again = load_file(os.path.join(out, "anima_init.safetensors"))
        for k in sd:
            self.assertTrue(torch.allclose(sd[k].float(), again[k].float()), k)  # lr 0 keeps the loaded weights


class MossDownloadTests(unittest.TestCase):
    def test_download_complete(self):
        from venno.models.moss.adapter import download_complete

        root = tempfile.mkdtemp(prefix="venno_dl_")
        self.addCleanup(shutil.rmtree, root, True)
        files = {
            "model_index.json": "{}",
            "tokenizer/tokenizer.json": "{}",
            "tokenizer/tokenizer_config.json": "{}",
            "transformer/config.json": "{}",
            "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                                "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "",
            "text_encoder/config.json": "{}",
            "text_encoder/model.safetensors": "",
            "vae/config.json": "{}",
            "vae/diffusion_pytorch_model.safetensors": "",
        }
        for name, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(root, name)), exist_ok=True)
            with open(os.path.join(root, name), "w") as f:
                f.write(text)
        self.assertTrue(download_complete(root))
        # an interrupted download leaves the shard as a hidden *.incomplete file
        os.remove(os.path.join(root, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"))
        self.assertFalse(download_complete(root))
        self.assertFalse(download_complete(os.path.join(root, "missing")))


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

    def test_report_timesteps_and_clean_share_schedule(self):
        out = os.path.join(self.tmp, "out_ts")
        job = F.base_job("moss_image", [{"path": os.path.join(self.tmp, "data", "a")}],
                         duration_unit="epochs", max_epochs=4, save_every_epochs=10, optimizer="AdamW", lr="1e-3",
                         output_name="moss_ts", workers=1, clean_end_share="60:1:30:2:2")
        job["dataset"].update(resolutions="64", batch_sizes="1", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["network"].update(dim=4, alpha=4)
        job["sampling"]["enabled"] = False
        self.assertEqual(job["training"]["timestep"], "moss_report")
        self.assertEqual(job["network"]["ignore_layers"], "", "all 42 blocks by default")
        lines, lrs = [], []
        orig_log, orig_step = T.log, torch.optim.AdamW.step
        T.log = lambda *a: (lines.append(" ".join(map(str, a))), orig_log(*a))

        def step(opt, *a, **k):
            lrs.append(opt.param_groups[0]["lr"])
            return orig_step(opt, *a, **k)

        torch.optim.AdamW.step = step
        try:
            trainer = T.Trainer(run_config(job, self.paths, out, os.path.join(self.tmp, "cache")))
            T._stop_requested = False
            trainer.train()
        finally:
            T.log, torch.optim.AdamW.step = orig_log, orig_step
        shares = [l for l in lines if l.startswith("Timesteps from epoch")]
        self.assertEqual(shares, [
            "Timesteps from epoch 1: 60% of steps at the clean end (noise below 50%)",
            "Timesteps from epoch 2: 30% of steps at the clean end (noise below 50%)",
            "Timesteps from epoch 3: 60% of steps at the clean end (noise below 50%)",
            "Timesteps from epoch 4: 30% of steps at the clean end (noise below 50%), at half the learning rate",
        ])
        per_epoch = len(lrs) // 4
        self.assertEqual(set(lrs[:3 * per_epoch]), {1e-3})
        self.assertEqual(set(lrs[3 * per_epoch:]), {5e-4}, "the refresh epoch trains at half the LR")
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-3, "the scheduler's LR is restored")
        g = torch.Generator().manual_seed(0)
        trainer.set_epoch(1)
        s = trainer.sample_sigmas(20000, g)
        self.assertAlmostEqual(float((s < 0.5).float().mean()), 0.30, delta=0.02)

    def test_samples_from_a_saved_lora(self):
        out = os.path.join(self.tmp, "out_fl")
        job = F.base_job("moss_image", [{"path": os.path.join(self.tmp, "data", "a")}],
                         duration_unit="steps", max_steps=4, save_every_epochs=10, optimizer="AdamW", lr="1e-2",
                         output_name="moss_fl", workers=1, ema_decay=0)
        job["dataset"].update(resolutions="64", batch_sizes="1", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["network"].update(dim=4, alpha=4)
        job["sampling"].update(enabled=False, width=64, height=64, steps=2,
                               prompts=[{"text": "A photo of a red object.", "enabled": True}])
        cache = os.path.join(self.tmp, "cache_fl")
        T._stop_requested = False
        T.Trainer(run_config(job, self.paths, out, cache)).train()
        lora = os.path.join(out, "moss_fl.safetensors")
        run = dict(run_config(job, self.paths, out, cache),
                   sample_lora={"lora": lora, "variants": [{}, {"drop_layers": "layers.0."}, {"uncond_skip": False}]})
        trainer = T.Trainer(run)
        skips, orig = [], trainer.adapter.sample

        def sample(*a, **k):
            skips.append(k.get("uncond_skip", "default"))
            return orig(*a, **k)

        trainer.adapter.sample = sample
        trainer.sample_lora(run["sample_lora"])
        names = sorted(f for f in os.listdir(os.path.join(out, "samples")) if f.endswith(".png"))
        self.assertEqual(names, ["moss_fl_00_42.png", "moss_fl_full-negative_00_42.png", "moss_fl_without-layers0_00_42.png"])
        self.assertEqual(skips, ["default", "default", ()], "only the third set runs every block in the negative branch")
        load = lambda n: np.asarray(Image.open(os.path.join(out, "samples", n)), dtype=np.int16)
        self.assertGreater(np.abs(load(names[0]) - load(names[2])).max(), 0, "muting block 0 of the LoRA changes the image")

    def test_caption_mix_caches_every_form(self):
        data = os.path.join(self.tmp, "data", "mix")
        F.make_images(data, 2, caption=lambda i: f"fox, solo, tag{i}\nA red fox number {i} sits in the grass.")
        out = os.path.join(self.tmp, "out_mix")
        job = F.base_job("moss_image", [{"path": data, "caption_mix": "25:25:50", "shuffle": True, "keep_tokens": 1}],
                         duration_unit="steps", max_steps=2, save_every_epochs=10, output_name="moss_mix", workers=1, te_variants=4, optimizer="AdamW")
        job["dataset"].update(resolutions="64", batch_sizes="1", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["network"].update(dim=4, alpha=4)
        job["sampling"]["enabled"] = False
        lines = []
        orig_log = T.log
        T.log = lambda *a: (lines.append(" ".join(map(str, a))), orig_log(*a))
        try:
            trainer = T.Trainer(run_config(job, self.paths, out, os.path.join(self.tmp, "cache_mix")))
            T._stop_requested = False
            trainer.train()
        finally:
            T.log = orig_log
        self.assertIn("  mix: caption mix tags 25%, prose 25%, both 50% for 2 image(s) with a caption line", lines)
        item = trainer.items[0]
        forms = trainer.caption_choices(item)
        self.assertEqual([len(t) for _, t in forms], [1, 1, 3], "4 variants shared by tags and both, plus the prose")
        for _, texts in forms:
            for text in texts:
                self.assertTrue(trainer.texts.has(text), text)
        self.assertEqual(forms[1][1], ["A red fox number 0 sits in the grass."])
        self.assertTrue(all(t.startswith("fox, ") and t.endswith(". A red fox number 0 sits in the grass.") for t in forms[2][1]))

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

    def test_batched_caption_encoding_matches_single(self):
        run = run_config(F.base_job("moss_image", [{"path": "/x"}]), self.paths, os.path.join(self.tmp, "o"), os.path.join(self.tmp, "c"))
        a = T.Trainer(run).adapter
        a.load_text_encoder()
        texts = ["a red cat", "a blue dog", "a big red cat", "a green car", "a red cat sitting on a mat"]
        lengths = [a.text_sort_key(t) for t in texts]
        self.assertGreater(len(lengths), len(set(lengths)))  # some share a length, so they batch
        batched = a.encode_text(texts)
        single = [a.encode_text([t], max_batch=1)[0] for t in texts]
        for b, s1 in zip(batched, single):
            self.assertEqual(b["embeds"].shape, s1["embeds"].shape)
            self.assertLess((b["embeds"].float() - s1["embeds"].float()).abs().max().item(), 1e-2)

    def test_compile_and_checkpoint_skip(self):
        out = os.path.join(self.tmp, "out_compile")
        job = F.base_job("moss_image", [{"path": os.path.join(self.tmp, "data", "a")}],
                         duration_unit="steps", max_steps=3, save_every_steps=10, optimizer="AdamW", lr="1e-3",
                         output_name="moss_compiled", workers=1, compile=True, ckpt_skip_blocks=1)
        job["dataset"].update(resolutions="64", batch_sizes="1", min_bucket=32, max_bucket=128, bucket_steps=16)
        job["network"].update(dim=4, alpha=4)
        job["sampling"]["enabled"] = False
        lines = []
        orig = T.log
        T.log = lambda *a: (lines.append(" ".join(map(str, a))), orig(*a))
        try:
            train(run_config(job, self.paths, out, os.path.join(self.tmp, "cache")))
        finally:
            T.log = orig
        self.assertTrue(any("torch.compile enabled" in l for l in lines), lines)
        self.assertTrue(any("checkpointing on 2 blocks (1 run without it)" in l for l in lines), lines)
        sd = load_file(os.path.join(out, "moss_compiled.safetensors"))
        self.assertGreater(sd["diffusion_model.layers.0.proj.q.lora_B.weight"].abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
