"""LoRA merges and compression reports (venno.merge) against dense matrix maths."""

import os
import shutil
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from venno import merge as M  # noqa: E402
from venno.lora import FrozenLoRA, LoRANetwork  # noqa: E402
from venno.models.moss.adapter import MossAdapter  # noqa: E402
from venno.models.moss.transformer import MossImageParams, MossImageTransformer  # noqa: E402


def deltas(path, ratio=1.0):
    """{module: dense change} of a LoRA file."""
    return {n: B @ A for n, (B, A) in M.LoraFile(path).factors(ratio).items()}


def best_rank(W, k):
    u, s, vh = torch.linalg.svd(W, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k]


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="venno_merge_")
        torch.manual_seed(0)
        cls.dit = MossImageTransformer(MossImageParams(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=32, freq_dim=32))
        cls.adapter = types.SimpleNamespace(dit=cls.dit)
        cls.targets = MossAdapter.lora_targets(cls.adapter, [])
        cls.a = cls.save_lora("a.safetensors", 6, 6)
        cls.b = cls.save_lora("b.safetensors", 3, 1.5)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def save_lora(cls, name, rank, alpha):
        net = LoRANetwork(cls.targets, rank, alpha)
        for lora in net.loras.values():
            torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
        path = os.path.join(cls.tmp, name)
        save_file({k: v.contiguous() for k, v in MossAdapter.export_lora(None, net.internal_state_dict()).items()}, path)
        return path

    def out(self, name):
        return os.path.join(self.tmp, name)

    def test_report(self):
        r = M.report([self.a])["files"][0]
        self.assertEqual((r["rank"], r["layout"]), (6, "peft"))
        self.assertEqual(r["modules"], 2 * 4, "q/k/v count as one fused module, as MOSS trains them")
        kept = [row["kept"] for row in r["rows"]]
        self.assertEqual([row["rank"] for row in r["rows"]], [6])  # only ranks below the file's own, and its own
        self.assertAlmostEqual(kept[-1], 1.0, places=9)
        r2 = M.report([self.a], ranks=(1, 2, 4))["files"][0]
        kept = [row["kept"] for row in r2["rows"]]
        self.assertEqual([row["rank"] for row in r2["rows"]], [1, 2, 4, 6])
        self.assertTrue(all(a < b for a, b in zip(kept, kept[1:])), kept)
        self.assertTrue(all(row["worst"] <= row["kept"] + 1e-12 for row in r2["rows"]))

    def test_concat_keeps_the_second_lora_whole(self):
        a, b = deltas(self.a), deltas(self.b, 1.2)
        res = M.merge([(self.a, 1.0), (self.b, 1.2)], self.out("exact.safetensors"), "concat", rank=9)
        self.assertEqual((res["rank"], res["first_rank"]), (9, 6))
        got = deltas(res["out"])
        for n in a:
            self.assertTrue(torch.allclose(got[n], a[n] + b[n], atol=1e-6), n)
        res = M.merge([(self.a, 1.0), (self.b, 1.2)], self.out("cut.safetensors"), "concat", rank=7)
        self.assertEqual(res["first_rank"], 4)
        self.assertLess(res["kept"], 1.0)
        got = deltas(res["out"])
        f = M.LoraFile(self.a)
        for unit, members in M.group_qkv(list(f.modules), [f]).items():
            dense_a = torch.cat([a[m] for m in members], 0)
            dense_b = torch.cat([b[m] for m in members], 0)
            want = best_rank(dense_a, 4) + dense_b  # the first compressed as well as possible, the second untouched
            self.assertTrue(torch.allclose(torch.cat([got[m] for m in members], 0), want, atol=1e-6), unit)

    def test_svd_mode(self):
        a, b = deltas(self.a), deltas(self.b)
        res = M.merge([(self.a, 1.0), (self.b, 1.0)], self.out("svd_exact.safetensors"), "svd", rank=9)
        got = deltas(res["out"])
        for n in a:
            self.assertTrue(torch.allclose(got[n], a[n] + b[n], atol=1e-6), n)
        res = M.merge([(self.a, 1.0), (self.b, 1.0)], self.out("svd_cut.safetensors"), "svd", rank=5)
        got = deltas(res["out"])
        f = M.LoraFile(self.a)
        for unit, members in M.group_qkv(list(f.modules), [f]).items():
            dense = torch.cat([a[m] + b[m] for m in members], 0)
            self.assertTrue(torch.allclose(torch.cat([got[m] for m in members], 0), best_rank(dense, 5), atol=1e-6), unit)

    def test_merged_file_loads_back_into_venno(self):
        res = M.merge([(self.a, 1.0), (self.b, 1.0)], self.out("load.safetensors"), "concat", rank=8)
        sd = load_file(res["out"])
        qs = sorted(k for k in sd if k.endswith(".proj.q.lora_A.weight"))
        self.assertTrue(qs)
        for q in qs:
            self.assertTrue(torch.equal(sd[q], sd[q.replace(".q.", ".k.")]) and torch.equal(sd[q], sd[q.replace(".q.", ".v.")]))
        internal = MossAdapter.import_lora(None, sd)  # refuses independent q/k/v downs
        net = LoRANetwork(self.targets, 8, 8)
        self.assertEqual(net.load_internal_state_dict(internal), [])
        frozen = FrozenLoRA(self.targets, internal)
        self.assertEqual(len(frozen.loras), len(self.targets))

    def test_kohya_layout(self):
        torch.manual_seed(1)
        sd = {}
        for i in range(3):
            sd[f"lora_unet_blocks_{i}_mlp_layer1.lora_down.weight"] = torch.randn(4, 16)
            sd[f"lora_unet_blocks_{i}_mlp_layer1.lora_up.weight"] = torch.randn(24, 4)
            sd[f"lora_unet_blocks_{i}_mlp_layer1.alpha"] = torch.tensor(2.0)
        path = self.out("kohya.safetensors")
        save_file(sd, path)
        res = M.merge([(path, 1.0), (path, 0.5)], self.out("kohya_merged.safetensors"), "svd", rank=4)
        got = load_file(res["out"])
        self.assertEqual(sorted(got), sorted(sd))
        self.assertEqual(got["lora_unet_blocks_0_mlp_layer1.lora_down.weight"].shape, (4, 16))
        want = deltas(path, 1.5)
        for n, W in deltas(res["out"]).items():
            self.assertTrue(torch.allclose(W, want[n], atol=1e-5), n)

    def test_cli(self):
        import contextlib
        import io
        import json

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = M.main(["merge", "--out", self.out("cli.safetensors"), "--mode", "concat", "--rank", "9",
                           f"{self.a}:1", f"{self.b}:0.5"])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertEqual(json.loads(buf.getvalue())["rank"], 9)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = M.main(["merge", "--out", self.out("bad.safetensors"), "--rank", "2", self.a, self.b])
        self.assertEqual(code, 1)
        self.assertIn("leaves no room", json.loads(buf.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
