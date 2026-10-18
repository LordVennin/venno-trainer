const test = require("node:test");
const assert = require("node:assert");
const { parseProgress } = require("../lib/runner");

test("parses sd-scripts and ai-toolkit tqdm lines", () => {
  const a = parseProgress("steps:  12%|█▏        | 120/1000 [01:23<10:00,  1.20it/s, avr_loss=0.0912]");
  assert.deepStrictEqual(a, { step: 120, total: 1000, elapsed: "01:23", eta: "10:00", loss: 0.0912 });
  const b = parseProgress("my_lora:  10%|█         | 300/3000 [05:00<45:00,  1.00s/it, lr: 1.0e-04 loss: 3.142e-01]");
  assert.strictEqual(b.step, 300);
  assert.strictEqual(b.total, 3000);
  assert.strictEqual(b.loss, 0.3142);
  assert.strictEqual(parseProgress("Caching latents: 50%|█ | 5/10 [00:01<00:01, 5.0it/s]"), null);
});
