"""cosine_settle: the curve of resuming one epoch at a time with Max Epochs raised by 1, in one run."""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from venno.optim import create_scheduler  # noqa: E402

SPE = 50  # steps per epoch


def lam(name, total, warmup=0, **kw):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return create_scheduler(name, opt, total, warmup, **kw).lr_lambdas[0]


class CosineSettleTests(unittest.TestCase):
    def test_matches_resuming_one_epoch_at_a_time(self):
        for warmup in (0, 20):
            settle = lam("cosine_settle", 12 * SPE, warmup, settle_from=4, steps_per_epoch=SPE)
            first = lam("cosine", 4 * SPE, warmup)  # the first run: Max Epochs 4
            for s in range(4 * SPE):
                self.assertAlmostEqual(settle(s), first(s), places=12)
            for k in range(5, 13):  # then each resume with Max Epochs k
                manual = lam("cosine", k * SPE, warmup)
                for s in range((k - 1) * SPE, k * SPE):
                    self.assertAlmostEqual(settle(s), manual(s), places=12, msg=f"epoch {k}, step {s}, warmup {warmup}")

    def test_max_epochs_does_not_change_the_curve(self):
        short = lam("cosine_settle", 8 * SPE, settle_from=4, steps_per_epoch=SPE)
        long = lam("cosine_settle", 30 * SPE, settle_from=4, steps_per_epoch=SPE)
        for s in range(8 * SPE):
            self.assertEqual(short(s), long(s))

    def test_every_epoch_settles_and_the_peaks_fall(self):
        f = lam("cosine_settle", 10 * SPE, settle_from=3, steps_per_epoch=SPE)
        self.assertEqual(f(0), 1.0)
        peaks = [f(e * SPE) for e in range(3, 10)]
        self.assertTrue(all(a > b for a, b in zip(peaks, peaks[1:])), peaks)
        self.assertAlmostEqual(peaks[0], 0.5 * (1 + math.cos(math.pi * 3 / 4)), places=9)  # epoch 4 starts 3/4 into a cosine over 4
        for e in range(1, 11):
            if e >= 3:
                self.assertLess(f(e * SPE - 1), 0.002, f"epoch {e} ends near 0")


if __name__ == "__main__":
    unittest.main()
