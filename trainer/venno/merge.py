"""Merge LoRA files and report how well they compress, on the CPU (no model needed).

Every LoRA module's change to its layer is a product up @ down * alpha / rank. The rank a merge keeps
is chosen from that product's singular values, computed from the factors (QR of each side, then the
SVD of a small R x R matrix), so even a 16384 x 3072 layer costs a fraction of a second.

  report: for each file, the share of its change kept at a range of ranks (the sum of squared singular
          values kept, over all modules), and the module that keeps least.
  merge, mode "concat": the first LoRA compressed to (rank - the others' ranks), the others kept whole,
          their factors stacked side by side. The result is exactly first(compressed) + others: only
          the first LoRA's weakest directions are dropped.
  merge, mode "svd": the LoRAs added together, then that sum compressed to the rank. Exact when the
          rank holds every input's rank; below that it keeps the strongest directions of the sum.

q/k/v modules that share their down matrix (MOSS-Image's fused attention, as Venno and AI-Toolkit
save it) are merged as one module and written back sharing it, so the result loads anywhere the
inputs did. Output keys follow the first file's layout (lora_A/lora_B or lora_down/lora_up); every
module gets the same rank, alpha = rank.

  python -m venno.merge report A.safetensors [B.safetensors ...]
  python -m venno.merge merge --out OUT.safetensors --mode concat --rank 128 A.safetensors:1 B.safetensors:1.2
Both print one JSON object."""

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUFFIXES = {
    "kohya": ("lora_down.weight", "lora_up.weight"),
    "peft": ("lora_A.weight", "lora_B.weight"),
}
REPORT_RANKS = (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 256)


class LoraFile:
    """A LoRA file as modules {name: (down, up, alpha)} plus its key layout, metadata and dtype."""

    def __init__(self, path):
        self.path = path
        with safe_open(path, "pt") as f:
            self.metadata = dict(f.metadata() or {})
            sd = {k: f.get_tensor(k) for k in f.keys()}
        self.style = None
        self.modules = OrderedDict()
        self.ignored = []
        for key in sd:
            for style, (down_s, up_s) in SUFFIXES.items():
                if key.endswith("." + down_s):
                    name = key[: -len(down_s) - 1]
                    up = sd.get(f"{name}.{up_s}")
                    if up is None:
                        raise ValueError(f"{path}: {key} has no {up_s}")
                    down = sd[key]
                    alpha = sd.get(f"{name}.alpha")
                    rank = down.shape[0]
                    self.modules[name] = (down, up, float(alpha) if alpha is not None else float(rank))
                    self.style = self.style or style
        known = {f"{n}.{s}" for n in self.modules for s in SUFFIXES[self.style or "kohya"]} | {f"{n}.alpha" for n in self.modules}
        self.ignored = sorted(k for k in sd if k not in known)
        if not self.modules:
            raise ValueError(f"{path}: no LoRA modules (lora_down/lora_up or lora_A/lora_B)")
        self.dtype = next(iter(self.modules.values()))[0].dtype
        self.rank = max(d.shape[0] for d, _, _ in self.modules.values())

    def factors(self, ratio=1.0):
        """{name: (B, A)} with B = up * alpha / rank * ratio (out x r) and A = down (r x in), in float64."""
        return {n: (up.double() * (alpha / down.shape[0]) * ratio, down.double()) for n, (down, up, alpha) in self.modules.items()}


def group_qkv(names, files):
    """Units to merge: {unit: [module names]}. q/k/v siblings become one unit when every file that has them
    shares one down matrix across the three (fused attention); everything else is its own unit."""
    units = OrderedDict()
    done = set()
    for name in names:
        if name in done:
            continue
        head, _, last = name.rpartition(".")
        trio = [f"{head}.{p}" for p in "qkv"]
        if head and last in "qkv" and all(t in names for t in trio):
            shared = True
            for f in files:
                present = [t for t in trio if t in f.modules]
                if present and (len(present) < 3 or not all(torch.equal(f.modules[t][0], f.modules[trio[0]][0]) for t in trio)):
                    shared = False
            if shared:
                units[f"{head}.qkv"] = trio
                done.update(trio)
                continue
        units[name] = [name]
        done.add(name)
    return units


def unit_factors(f, members, ratio=1.0):
    """(B, A) of one unit of one file, or None if the file lacks it: q/k/v ups stacked over the shared down."""
    fac = f.factors(ratio)
    if not all(m in fac for m in members):
        return None
    if len(members) == 1:
        return fac[members[0]]
    return torch.cat([fac[m][0] for m in members], 0), fac[members[0]][1]


def svd_factors(B, A):
    """Singular values and vectors of B @ A from its factors: (U * S) and Vh with B @ A = U diag(S) Vh."""
    qb, rb = torch.linalg.qr(B)
    qa, ra = torch.linalg.qr(A.T)
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    return qb @ u, s, vh @ qa.T


def truncate(B, A, k):
    """The best rank-k factors of B @ A (Eckart-Young), as (B', A') with the singular values split between them."""
    u, s, vh = svd_factors(B, A)
    k = min(k, s.shape[0])
    root = s[:k].sqrt()
    return u[:, :k] * root, root[:, None] * vh[:k], s


def kept_share(s, k):
    total = float((s ** 2).sum())
    return 1.0 if total == 0 else float((s[:k] ** 2).sum()) / total


def report(paths, ranks=REPORT_RANKS):
    out = []
    for path in paths:
        f = LoraFile(path)
        units = group_qkv(list(f.modules), [f])
        spectra = {u: svd_factors(*unit_factors(f, m))[1] for u, m in units.items()}
        total = sum(float((s ** 2).sum()) for s in spectra.values())
        rows = []
        for k in sorted({r for r in ranks if r < f.rank} | {f.rank}):
            kept = sum(float((s[:k] ** 2).sum()) for s in spectra.values())
            worst = min(spectra, key=lambda u: kept_share(spectra[u], k))
            rows.append({"rank": k, "kept": kept / total if total else 1.0,
                         "worst": kept_share(spectra[worst], k), "worst_module": worst})
        out.append({"path": path, "rank": f.rank, "modules": len(units), "layout": f.style,
                    "size_mb": os.path.getsize(path) / 2 ** 20, "rows": rows})
    return {"files": out}


def merge(specs, out_path, mode="concat", rank=None, dtype=None):
    """specs: [(path, ratio)], the first being the LoRA to compress in concat mode."""
    files = [(LoraFile(p), r) for p, r in specs]
    first = files[0][0]
    names = list(OrderedDict.fromkeys(n for f, _ in files for n in f.modules))
    units = group_qkv(names, [f for f, _ in files])
    others_rank = sum(f.rank for f, _ in files[1:])
    rank = int(rank or first.rank)
    if mode == "concat":
        keep = rank - others_rank
        if keep < 1:
            raise ValueError(f"Rank {rank} leaves no room for the first LoRA: the others need {others_rank}")
    elif mode != "svd":
        raise ValueError(f"Unknown merge mode {mode}")
    merged, kept, worst = OrderedDict(), [0.0, 0.0], [1.0, None]

    def note(s, k, unit):
        kept[0] += float((s[:k] ** 2).sum())
        kept[1] += float((s ** 2).sum())
        share = kept_share(s, k)
        if worst[1] is None or share < worst[0]:
            worst[:] = [share, unit]

    for unit, members in units.items():
        parts = [(i, unit_factors(f, members, ratio)) for i, (f, ratio) in enumerate(files)]
        parts = [(i, p) for i, p in parts if p is not None]
        if mode == "concat":
            Bs, As = [], []
            for i, (B, A) in parts:
                if i == 0:
                    B, A, s = truncate(B, A, keep)
                    note(s, keep, unit)
                Bs.append(B)
                As.append(A)
            B, A = torch.cat(Bs, 1), torch.cat(As, 0)
        else:
            B = torch.cat([p[0] for _, p in parts], 1)
            A = torch.cat([p[1] for _, p in parts], 0)
            B, A, s = truncate(B, A, rank)
            note(s, rank, unit)
        if B.shape[1] < rank:  # every module at the same rank: pad with zero directions
            pad = rank - B.shape[1]
            B = torch.cat([B, B.new_zeros(B.shape[0], pad)], 1)
            A = torch.cat([A, A.new_zeros(pad, A.shape[1])], 0)
        merged[unit] = (members, B, A)

    down_s, up_s = SUFFIXES[first.style]
    out_dtype = dtype or first.dtype
    sd = {}
    for unit, (members, B, A) in merged.items():
        ups = list(B.split([first.modules[m][1].shape[0] if m in first.modules else
                            next(f.modules[m][1].shape[0] for f, _ in files if m in f.modules) for m in members], 0))
        for m, up in zip(members, ups):
            sd[f"{m}.{down_s}"] = A.to(out_dtype).contiguous()
            sd[f"{m}.{up_s}"] = up.to(out_dtype).contiguous()
            sd[f"{m}.alpha"] = torch.tensor(float(rank))
    meta = dict(first.metadata)
    meta.update({"ss_network_dim": str(rank), "ss_network_alpha": str(rank),
                 "venno_merge": json.dumps({"mode": mode, "rank": rank,
                                            "inputs": [[os.path.basename(f.path), r] for f, r in files]})})
    tmp = out_path + ".tmp"
    save_file(sd, tmp, metadata=meta)
    os.replace(tmp, out_path)
    share = kept[0] / kept[1] if kept[1] else 1.0
    return {"out": out_path, "rank": rank, "mode": mode, "size_mb": os.path.getsize(out_path) / 2 ** 20,
            "first_rank": keep if mode == "concat" else None,
            "kept": share, "worst": worst[0], "worst_module": worst[1],
            "kept_of": "the first LoRA" if mode == "concat" else "the LoRAs' sum"}


def parse_spec(spec):
    """'PATH' or 'PATH:RATIO' -> (path, ratio)."""
    path, sep, ratio = spec.rpartition(":")
    try:
        return (path, float(ratio)) if sep and path else (spec, 1.0)
    except ValueError:
        return spec, 1.0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m venno.merge")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("files", nargs="+")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--mode", choices=("concat", "svd"), default="concat")
    m.add_argument("--rank", type=int)
    m.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    m.add_argument("loras", nargs="+", help="PATH or PATH:RATIO; the first is compressed in concat mode")
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    try:
        if a.cmd == "report":
            res = report(a.files)
        else:
            specs = [parse_spec(s) for s in a.loras]
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(a.dtype)
            res = merge(specs, a.out, a.mode, a.rank, dtype)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
