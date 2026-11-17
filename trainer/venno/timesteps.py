"""MOSS-Image's own training timesteps, and the Clean-end share that moves them.

The distribution is the final pretraining interval of the MOSS-Image technical
report (Appendix A.2, Table 6: the stage the released weights finished on):
with probability 0.01 a uniform pick of the 1000-step grid, otherwise
sigma = 1 - sigmoid(Z), Z ~ N(mu, 0.95^2), truncated to sigma <= 0.95,
rescaled by 1 / 0.95 and snapped to the grid. Its mu of -1.09 puts about 11%
of steps at the clean end (sigma < 0.5), where fine detail and identity are
learned; AI-Toolkit's shift-5 uniform timesteps put about 17% there.

The Clean-end share moves mu so a chosen share of steps lands below sigma 0.5,
keeping the curve's shape, optionally changing over the run:
    "40"            40% every epoch
    "75:5:40"       75% for epochs 1-5, then 40%
    "60:7:30:8"     60% until epoch 7, sliding in a straight line to 30% by epoch 8
    "60:7:30:8:3"   ... and after epoch 8 the higher share, with every 3rd epoch
                    (11, 14, 17, ...) at the lower one and half the learning rate
    "25@1, 30@6, 45@10, 50@15"   keyframes, as many as wanted: share@epoch, a straight
                    slide between them, held before the first and after the last
Ported from Fizgig (github.com/shootthesound/Fizgig, moss_image/driver.py,
Apache-2.0), whose default for MOSS-Image LoRAs is the last one ("Slide")."""

import math

import torch

T_MU, T_S, T_CUTOFF, T_UNIFORM = -1.09, 0.95, 0.95, 0.01
TRAIN_STEPS = 1000


def _cdf(x, mu):
    """P(sigma <= x) for sigma = 1 - sigmoid(Z), Z ~ N(mu, T_S^2)."""
    return 1.0 - 0.5 * math.erfc(-((math.log((1 - x) / x) - mu) / T_S) / math.sqrt(2))


def clean_share(mu):
    """Share of steps below sigma 0.5 for a given mu, the 1% uniform included."""
    return (1.0 - T_UNIFORM) * _cdf(0.5 * T_CUTOFF, mu) / _cdf(T_CUTOFF, mu) + T_UNIFORM * 0.5


def mu_for_clean_share(pct):
    """The mu that puts `pct`% of steps at the clean end (bisection; the share rises with mu)."""
    want = (float(pct) / 100.0 - T_UNIFORM * 0.5) / (1.0 - T_UNIFORM)
    want = min(max(want, 1e-4), 1.0 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _cdf(0.5 * T_CUTOFF, mid) / _cdf(T_CUTOFF, mid) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


SHARE_FORMATS = ("give a share (40), share:epochs:share (75:5:40 = 75% for the first 5 epochs, then 40%), "
                 "share:epoch:share:epoch[:every] (60:7:30:8:3 = 60% until epoch 7, 30% by epoch 8, then every 3rd "
                 "epoch at 30% and half the learning rate, the others at 60%), or keyframes share@epoch (25@1, 30@6, "
                 "45@10 = a straight slide between them)")


def parse_keyframes(spec, setting="Clean-end share", value="share", high=100.0, example="30@6"):
    """'25@1, 30@6, 45@10' -> [(1, 25.0), (6, 30.0), (10, 45.0)], epochs rising, values above 0 and up to
    `high`; ValueError naming the setting if malformed."""
    frames = []
    for item in [x for x in str(spec).replace("%", "").replace(";", ",").replace(",", " ").split() if x]:
        v, at, epoch = item.partition("@")
        try:
            frames.append((int(epoch), float(v)))
        except ValueError:
            raise ValueError(f"{setting} '{spec}': '{item}' is not {value}@epoch (e.g. {example})")
    if not frames:
        raise ValueError(f"{setting} '{spec}': give {value}@epoch keyframes (e.g. {example})")
    for (e0, _), (e1, _) in zip(frames, frames[1:]):
        if e1 <= e0:
            raise ValueError(f"{setting} '{spec}': keyframe epochs must rise ({e0} then {e1})")
    for e, v in frames:
        if e < 1 or not (0 < v <= high):
            raise ValueError(f"{setting} '{spec}': {v:g}@{e} needs a {value} above 0 and up to {high:g}, at epoch 1 or later")
    return frames


def interpolate(frames, epoch):
    """The value at 1-based `epoch`: a straight line between keyframes, held before the first and after the last."""
    e = int(epoch)
    if e <= frames[0][0]:
        return frames[0][1]
    for (e0, v0), (e1, v1) in zip(frames, frames[1:]):
        if e <= e1:
            return v0 + (v1 - v0) * (e - e0) / (e1 - e0)
    return frames[-1][1]


def shift_for_epoch(spec, epoch):
    """Flow Shift for 1-based `epoch`: a number (blank = 1, no shift), or keyframes shift@epoch such as
    '5@1, 5@6, 2@10' (5 through epoch 6, sliding to 2 by epoch 10)."""
    s = str(spec if spec is not None else "").strip()
    if "@" not in s:
        try:
            v = float(s) if s else 1.0
        except ValueError:
            raise ValueError(f"Flow Shift '{spec}': give a number (5) or keyframes shift@epoch (5@1, 5@6, 2@10)")
        if not v > 0:
            raise ValueError(f"Flow Shift '{spec}' must be above 0 (1 = no shift)")
        return v
    return interpolate(parse_keyframes(s, "Flow Shift", "shift", 100.0, "5@6"), epoch)


def parse_share(spec):
    """Validate a Clean-end share; returns its numbers ([] = the report's own schedule), or for keyframes a list
    of (epoch, share) pairs."""
    s = str(spec or "").replace("%", "").strip()
    if not s:
        return []
    if "@" in s:
        return parse_keyframes(s)
    try:
        parts = [float(x) for x in s.split(":")]
    except ValueError:
        parts = None
    if not parts or len(parts) not in (1, 3, 4, 5) or any(not (0 < p <= 100) for p in parts[::2]) \
            or (len(parts) >= 4 and parts[3] < parts[1]):
        raise ValueError(f"Clean-end share '{spec}': {SHARE_FORMATS}")
    return parts


def share_for_epoch(spec, epoch):
    """(share in % or None for the report's schedule, whether this epoch trains at half the LR) for 1-based `epoch`."""
    parts = parse_share(spec)
    if not parts:
        return None, False
    if isinstance(parts[0], tuple):  # keyframes
        return interpolate(parts, epoch), False
    if len(parts) == 1:
        return parts[0], False
    first, until, rest = parts[:3]
    end = parts[3] if len(parts) >= 4 else until
    every = int(parts[4]) if len(parts) == 5 else 0
    e = int(epoch)
    if e <= until:
        share = first
    elif e >= end:
        share = rest
    else:
        share = first + (rest - first) * (e - until) / (end - until)
    half = False
    if every > 0 and e > end:
        refresh = (e - end) % every == 0
        share = min(first, rest) if refresh else max(first, rest)
        half = first != rest and refresh
    return share, half


def sample_report_sigmas(n, generator, mu=T_MU):
    """n sigmas from the report's distribution (with its grid), centred on mu."""
    u = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    p = u[:, 1].clamp_min(1e-12) * _cdf(T_CUTOFF, mu)
    z = mu + T_S * torch.special.ndtri(1.0 - p)
    s = torch.sigmoid(-z)  # 1 - sigmoid(z)
    sigma = torch.ceil(TRAIN_STEPS * s / T_CUTOFF).clamp(1, TRAIN_STEPS) / TRAIN_STEPS
    uniform = (torch.floor(u[:, 1] * TRAIN_STEPS) + 1).clamp(max=TRAIN_STEPS) / TRAIN_STEPS
    return torch.where(u[:, 0] < T_UNIFORM, uniform, sigma).float()
