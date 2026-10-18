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


if __name__ == "__main__":
    unittest.main()
