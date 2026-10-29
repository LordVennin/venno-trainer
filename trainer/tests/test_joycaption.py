"""JoyCaption on CPU with a tiny random LLaVA in the real checkpoint layout
(the real processor, tokenizer and chat template, shrunk weights): batched
left-padded generation, decoding only new tokens, the service's GPU rule and
the shared model slot (a tagger request unloads JoyCaption)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json",
         "chat_template.json", "special_tokens_map.json", "generation_config.json"]


def make_tiny_joycaption(models_dir):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    from venno.tagger import joycaption as J

    root = J.model_dir(models_dir, "joycaption-beta-one")
    os.makedirs(root, exist_ok=True)
    for f in FILES:
        shutil.copy(hf_hub_download(REPO, f), root)
    with open(os.path.join(root, "config.json")) as f:
        cfg = json.load(f)
    cfg["text_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    cfg["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2)
    torch.manual_seed(0)
    config = LlavaConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "transformers_version")})
    LlavaForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(root)
    return root


class JoyCaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="venno_joy_")
        cls.models = os.path.join(cls.tmp, "models")
        try:
            cls.root = make_tiny_joycaption(cls.models)
        except Exception as e:  # offline
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest(f"JoyCaption processor files not downloadable: {e}")
        cls.images = []
        for i, size in enumerate(((300, 200), (90, 400))):
            p = os.path.join(cls.tmp, f"img{i}.png")
            Image.new("RGB", size, (40 * i, 120, 200)).save(p)
            cls.images.append(p)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_load_kwargs_without_accelerate(self):
        from venno.tagger import joycaption as J

        cuda = torch.device("cuda")
        self.assertEqual(J.load_kwargs(cuda, "bf16", accelerate=True)["device_map"], {"": "cuda"})
        self.assertNotIn("device_map", J.load_kwargs(cuda, "bf16", accelerate=False), "bf16 loads, then moves to the GPU")
        self.assertNotIn("device_map", J.load_kwargs(torch.device("cpu"), "bf16", accelerate=True))
        with self.assertRaisesRegex(RuntimeError, "needs the accelerate package"):
            J.load_kwargs(cuda, "nf4", accelerate=False)

    def test_generation(self):
        from venno.tagger import joycaption as J

        self.assertTrue(J.is_downloaded(self.models, "joycaption-beta-one"))
        cap = J.JoyCaptioner(self.root, "cpu")
        chat = cap.chat("Write a long caption.")
        self.assertIn("You are a helpful image captioner.", chat)
        self.assertIn("<|reserved_special_token_69|>", chat)  # the template inserts the image token
        self.assertTrue(chat.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        prompts = ["Write a short caption.", "Write a long descriptive caption for this image. These tags were written by a person and are accurate: fox, red fur."]
        greedy = dict(do_sample=False, max_new_tokens=10)
        one = [J.run(cap, [p], [q], **greedy)[0]["text"] for p, q in zip(self.images, prompts)]
        both = J.run(cap, self.images + [os.path.join(self.tmp, "missing.png")], prompts + ["x"], batch_size=2, **greedy)
        self.assertEqual([r["text"] for r in both[:2]], one, "left-padded batches match one-by-one")
        self.assertIn("error", both[2])
        for text in one:  # only new tokens: never the prompt
            self.assertNotIn("helpful image captioner", text)
            self.assertLessEqual(len(cap.processor.tokenizer(text, add_special_tokens=False)["input_ids"]), 12)
        sampled = J.run(cap, self.images[:1], prompts[:1], temperature=0.6, top_p=0.9, max_new_tokens=6)
        self.assertIn("text", sampled[0])

    def _service(self, reqs, allow_cpu):
        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE))
        env.pop("VENNO_CAPTION_ALLOW_CPU", None)
        if allow_cpu:
            env["VENNO_CAPTION_ALLOW_CPU"] = "1"
        proc = subprocess.run([sys.executable, "-m", "venno.tagger"], input="\n".join(json.dumps(r) for r in reqs) + "\n",
                              capture_output=True, text=True, env=env, timeout=600)
        return {m["id"]: m for m in (json.loads(l) for l in proc.stdout.splitlines() if l.strip()) if "id" in m}

    def test_service(self):
        import torch

        req = {"id": 1, "cmd": "caption", "model": "joycaption-beta-one", "paths": self.images, "prompts": ["a", "b"],
               "models_dir": self.models, "batch_size": 2, "gen": {"do_sample": False, "max_new_tokens": 5}}
        if not torch.cuda.is_available():
            out = self._service([req], allow_cpu=False)
            self.assertFalse(out[1]["ok"])
            self.assertIn("needs an NVIDIA GPU", out[1]["error"])
        reqs = [req]
        try:
            from tests.test_tagger import make_wd_model

            make_wd_model(self.models)
            reqs += [{"id": 3, "cmd": "tag", "model": "wd-vit-v3", "device": "cpu", "paths": self.images[:1], "models_dir": self.models}]
        except ImportError:  # no timm: skip the swap check
            pass
        out = self._service(reqs, allow_cpu=True)
        self.assertTrue(out[1]["ok"], out[1].get("error"))
        self.assertEqual(len(out[1]["results"]), 2)
        self.assertTrue(all("text" in r for r in out[1]["results"]))
        self.assertEqual(out[1]["loaded"], ["joycaption-beta-one", "cpu", "bf16"])
        if 3 in out:
            self.assertTrue(out[3]["ok"], out[3].get("error"))
            self.assertEqual(out[3]["loaded"], ["wd-vit-v3", "cpu"], "the tagger replaced JoyCaption in the model slot")


if __name__ == "__main__":
    unittest.main()
