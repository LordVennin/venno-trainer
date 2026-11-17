"""MOSS-Image report timesteps, the Clean-end share schedules and EMA weights."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from venno import timesteps as TS  # noqa: E402
from venno.ema import EMAWeights  # noqa: E402


class TimestepTests(unittest.TestCase):
    def test_report_distribution(self):
        self.assertAlmostEqual(100 * TS.clean_share(TS.T_MU), 11.2, delta=0.1)  # "about 11%"
        for pct in (11, 30, 60, 90):
            self.assertAlmostEqual(100 * TS.clean_share(TS.mu_for_clean_share(pct)), pct, places=2)
        s = TS.sample_report_sigmas(100000, torch.Generator().manual_seed(1), TS.mu_for_clean_share(60))
        self.assertAlmostEqual(float((s < 0.5).float().mean()), 0.60, delta=0.01)
        self.assertGreaterEqual(float(s.min()), 1 / 1000)
        self.assertLessEqual(float(s.max()), 1.0)
        self.assertTrue(torch.allclose(s * 1000, (s * 1000).round(), atol=1e-3), "on the 1000-step grid")

    def test_schedules(self):
        self.assertEqual(TS.share_for_epoch("", 3), (None, False))
        self.assertEqual(TS.share_for_epoch("40%", 9), (40, False))
        self.assertEqual([TS.share_for_epoch("75:5:40", e)[0] for e in (1, 5, 6, 20)], [75, 75, 40, 40])
        self.assertEqual([TS.share_for_epoch("60:4:30:8", e)[0] for e in (4, 6, 8, 9)], [60, 45, 30, 30])
        keys = "25@1, 25@4, 30@6, 40@10, 50@15"
        self.assertEqual([TS.share_for_epoch(keys, e)[0] for e in range(1, 17)],
                         [25, 25, 25, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42, 44, 46, 48, 50, 50])
        self.assertEqual([TS.share_for_epoch("25@5 40@6", e) for e in (1, 5, 6, 9)], [(25, False), (25, False), (40, False), (40, False)])
        self.assertEqual(TS.share_for_epoch("30%@3", 1)[0], 30, "held before the first keyframe")
        for bad in ("30@6, 25@4", "30@6, 40@6", "30@0", "120@3", "x@2", "30@", "30@2.5"):
            with self.assertRaises(ValueError, msg=bad):
                TS.parse_share(bad)
        self.assertEqual([TS.shift_for_epoch("5@1, 5@6, 2@10", e) for e in (1, 6, 7, 8, 10, 15)], [5, 5, 4.25, 3.5, 2, 2])
        self.assertEqual([TS.shift_for_epoch(v, 4) for v in (5.0, "3", "", None)], [5, 3, 1, 1])
        for bad in ("0", "-1", "abc", "5@6, 2@3", "0@2", "5@"):
            with self.assertRaises(ValueError, msg=bad):
                TS.shift_for_epoch(bad, 1)
        slide = [TS.share_for_epoch("60:7:30:8:3", e) for e in range(1, 18)]
        self.assertEqual([s for s, _ in slide], [60] * 7 + [30, 60, 60, 30, 60, 60, 30, 60, 60, 30])
        self.assertEqual([e + 1 for e, (_, half) in enumerate(slide) if half], [11, 14, 17])
        for bad in ("abc", "60:7", "0", "120", "60:7:30:5", "60:7:30:8:3:1"):
            with self.assertRaises(ValueError, msg=bad):
                TS.parse_share(bad)


class EMATests(unittest.TestCase):
    def test_average_swap_and_state(self):
        p = torch.nn.Parameter(torch.zeros(3))
        ema = EMAWeights([p], decay=0.5)
        expect = 0.0
        for n, v in enumerate((1.0, 2.0, 3.0), start=1):
            with torch.no_grad():
                p.fill_(v)
            ema.update()
            d = min(0.5, (1 + n) / (10 + n))  # the ramp: early steps track the weights
            expect = expect * d + v * (1 - d)
        self.assertTrue(torch.allclose(ema.shadow[0], torch.full((3,), expect)))
        flat = EMAWeights([torch.nn.Parameter(torch.zeros(1))], decay=0.5, ramp=0)
        for v in (1.0, 2.0, 3.0):
            with torch.no_grad():
                flat.params[0].fill_(v)
            flat.update()
        self.assertAlmostEqual(float(flat.shadow[0]), 2.125)  # 0.5 * prev + 0.5 * v
        with ema.applied():
            self.assertTrue(torch.allclose(p, torch.full((3,), expect)))
        self.assertTrue(torch.equal(p.detach(), torch.full((3,), 3.0)))
        other = EMAWeights([torch.nn.Parameter(torch.zeros(3))], decay=0.5)
        other.load_state_dict(ema.state_dict())
        self.assertEqual(other.n, 3)
        self.assertTrue(torch.equal(other.shadow[0], ema.shadow[0]))


if __name__ == "__main__":
    unittest.main()
