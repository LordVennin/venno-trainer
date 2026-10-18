"""Ported model code vs references."""

import os
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from venno.models.moss.adapter import moss_sigmas  # noqa: E402
from venno.models.moss.transformer import MossImageParams, MossImageTransformer, fuse_split_qkv  # noqa: E402


def load_vendor_moss():
    """Import the vendor pipeline file with the diffusers pieces it needs stubbed."""
    from torch import nn

    def register_to_config(fn):
        def wrapper(self, *args, **kwargs):
            import inspect

            names = list(inspect.signature(fn).parameters)[1:]
            cfg = {p.name: p.default for p in list(inspect.signature(fn).parameters.values())[1:]}
            cfg.update(dict(zip(names, args)))
            cfg.update(kwargs)
            fn(self, *args, **kwargs)
            self.config = types.SimpleNamespace(**cfg)
        return wrapper

    class Mixin(nn.Module):
        pass

    class Output:
        def __init__(self, sample=None, images=None):
            self.sample, self.images = sample, images

    stubs = {
        "diffusers": dict(AutoencoderKLFlux2=object, DiffusionPipeline=object, FlowMatchEulerDiscreteScheduler=object),
        "diffusers.schedulers": {},
        "diffusers.schedulers.scheduling_utils": dict(SchedulerMixin=object),
        "diffusers.configuration_utils": dict(ConfigMixin=object, register_to_config=register_to_config),
        "diffusers.image_processor": dict(VaeImageProcessor=object),
        "diffusers.models": {},
        "diffusers.models.modeling_utils": dict(ModelMixin=Mixin),
        "diffusers.models.modeling_outputs": dict(Transformer2DModelOutput=Output),
        "diffusers.pipelines": {},
        "diffusers.pipelines.pipeline_utils": dict(ImagePipelineOutput=Output),
        "diffusers.utils": {},
        "diffusers.utils.torch_utils": dict(randn_tensor=None),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    try:
        for name, attrs in stubs.items():
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            sys.modules[name] = mod
        import importlib.util

        spec = importlib.util.spec_from_file_location("moss_vendor", os.path.join(HERE, "fixtures", "moss_vendor_pipeline.py"))
        vendor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vendor)
        return vendor
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class MossEquivalence(unittest.TestCase):
    def test_transformer_matches_vendor(self):
        vendor = load_vendor_moss()
        torch.manual_seed(0)
        cfg = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=32, latent_channels=128)
        ref = vendor.MOSSImageTransformer(**cfg).double()
        with torch.no_grad():
            for p in ref.parameters():
                p.normal_(0, 0.05)
        # vendor checkpoint (split q/k/v, rope.bands) -> our fused layout
        sd = {k: v.clone() for k, v in ref.state_dict().items()}
        sd = MossImageTransformer.convert_state_dict_on_load(sd)
        ours = MossImageTransformer(MossImageTransformer.aitk_config_from_state_dict(sd)).double()
        ours.load_state_dict(sd, strict=True)

        x = torch.randn(2, 128, 3, 5, dtype=torch.float64)
        t = torch.tensor([700.0, 150.0])
        text = torch.randn(2, 6, 48, dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        with torch.no_grad():
            want = ref(x, t, text, mask, return_dict=False)[0]
            got = ours(x, t, text, mask)
            self.assertLess((want - got).abs().max().item(), 1e-9)
            want = ref(x, t, text, mask, return_dict=False, skip_layers=[1])[0]
            got = ours(x, t, text, mask, skip_layers=[1])
            self.assertLess((want - got).abs().max().item(), 1e-9)

    def test_sigmas_match_vendor_scheduler_grid(self):
        s = moss_sigmas(30, shift=5.0)
        self.assertAlmostEqual(float(s[0]), 1.0, places=6)
        self.assertEqual(float(s[-1]), 0.0)
        self.assertTrue(bool((s[1:] < s[:-1]).all()))


class CheckpointSkipTests(unittest.TestCase):
    """Partial gradient checkpointing must not change the gradients."""

    def grads(self, model, run, targets):
        from venno.lora import LoRANetwork

        torch.manual_seed(1)
        net = LoRANetwork(targets, 4, 4)
        with torch.no_grad():
            for lora in net.loras.values():
                lora.lora_up.weight.normal_(0, 0.1)
        net.apply_to()
        model.train()
        run().pow(2).mean().backward()
        net.remove()
        return [p.grad.clone() for p in net.parameters()]

    def test_moss(self):
        torch.manual_seed(0)
        m = MossImageTransformer(MossImageParams(dim=64, ffn_dim=96, num_heads=4, num_layers=4, text_dim=32, freq_dim=32))
        x, t, text = torch.randn(1, 128, 3, 4), torch.tensor([400.0]), torch.randn(1, 5, 32)
        targets = [(f"l{i}", layer.proj["qkv"]) for i, layer in enumerate(m.layers)]
        results = []
        for ckpt, skip in ((False, set()), (True, set()), (True, {2, 3})):
            m.gradient_checkpointing, m.ckpt_skip = ckpt, skip
            results.append(self.grads(m, lambda: m(x, t, text), targets))
        for other in results[1:]:
            for a, b in zip(results[0], other):
                self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_anima(self):
        from venno.models.anima import dit as D

        torch.manual_seed(0)
        m = D.MiniTrainDIT(
            max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2,
            patch_temporal=1, model_channels=128, num_blocks=3, num_heads=4, crossattn_emb_channels=1024,
            pos_emb_cls="rope3d", pos_emb_learnable=True, use_adaln_lora=True, adaln_lora_dim=32,
            rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_enable_fps_modulation=False,
        )
        with torch.no_grad():
            for blk in m.blocks:
                for seq in (blk.adaln_modulation_self_attn, blk.adaln_modulation_cross_attn, blk.adaln_modulation_mlp):
                    seq[2].weight.normal_(0, 0.05)
        x, t, ctx = torch.randn(1, 16, 1, 4, 4), torch.tensor([0.3]), torch.randn(1, 7, 1024)
        pad = torch.zeros(1, 1, 4, 4)
        targets = [(f"b{i}", blk.self_attn.q_proj) for i, blk in enumerate(m.blocks)]
        results = []
        for mode in ("none", "all", "skip"):
            for i, blk in enumerate(m.blocks):
                blk.gradient_checkpointing = mode == "all" or (mode == "skip" and i == 0)
            results.append(self.grads(m, lambda: m(x, t, ctx, padding_mask=pad), targets))
        for other in results[1:]:
            for a, b in zip(results[0], other):
                self.assertTrue(torch.allclose(a, b, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
