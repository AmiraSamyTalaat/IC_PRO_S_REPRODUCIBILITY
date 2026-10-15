"""Synthetic multi-instrument benchmark with explicit non-identifiability pairs."""
from __future__ import annotations
import math
import torch


def make_wavelength_grid(n=256, lo=0.0, hi=1.0, device=None):
    return torch.linspace(lo, hi, n, device=device)


def _normalize(x):
    return x.clamp_min(0) / x.amax(dim=-1, keepdim=True).clamp_min(1e-8)


def random_spectra(batch, wl, family="mixed", max_peaks=6):
    n = wl.numel()
    x = torch.zeros(batch, n, device=wl.device)
    for b in range(batch):
        fam = family
        if family == "random":
            fam = ["narrow", "doublet", "broad", "mixed", "absorption"][int(torch.randint(0, 5, (1,)))]

        if fam in ("narrow", "doublet", "mixed"):
            k = 1 if fam == "narrow" else (2 if fam == "doublet" else int(torch.randint(1, max_peaks + 1, (1,))))
            if fam == "doublet":
                c0 = torch.empty(1, device=wl.device).uniform_(0.15, 0.85)
                sep = torch.empty(1, device=wl.device).uniform_(0.008, 0.05)
                centers = torch.cat([(c0 - sep / 2).clamp(0.03, 0.97), (c0 + sep / 2).clamp(0.03, 0.97)])
            else:
                centers = torch.rand(k, device=wl.device) * 0.9 + 0.05
            widths = torch.exp(torch.empty(k, device=wl.device).uniform_(math.log(0.004), math.log(0.07)))
            amps = torch.empty(k, device=wl.device).uniform_(0.25, 1.0)
            for c, s, a in zip(centers, widths, amps):
                x[b] += a * torch.exp(-0.5 * ((wl - c) / s) ** 2)

        if fam in ("broad", "mixed", "absorption"):
            a0 = torch.empty(1, device=wl.device).uniform_(0.15, 0.45)
            a1 = torch.empty(1, device=wl.device).uniform_(-0.20, 0.20)
            a2 = torch.empty(1, device=wl.device).uniform_(-0.12, 0.12)
            base = a0 + a1 * (wl - 0.5) + a2 * (wl - 0.5) ** 2
            x[b] += base.clamp_min(0.03)

        if fam == "absorption":
            k = int(torch.randint(1, 4, (1,)))
            for _ in range(k):
                c = torch.empty(1, device=wl.device).uniform_(0.08, 0.92)
                s = torch.empty(1, device=wl.device).uniform_(0.01, 0.06)
                a = torch.empty(1, device=wl.device).uniform_(0.05, 0.25)
                x[b] -= a * torch.exp(-0.5 * ((wl - c) / s) ** 2)
    return _normalize(x)


def random_response(m, wl, complexity=5, family="train", device=None):
    device = device or wl.device
    n = wl.numel()
    R = torch.zeros(m, n, device=device)
    for i in range(m):
        k = complexity
        for _ in range(k):
            if family == "ood":
                c = torch.rand(1, device=device)
                s = torch.empty(1, device=device).uniform_(0.008, 0.10)
            else:
                c = torch.rand(1, device=device)
                s = torch.empty(1, device=device).uniform_(0.02, 0.20)
            a = torch.empty(1, device=device).uniform_(0.2, 1.0)
            R[i] += a * torch.exp(-0.5 * ((wl - c) / s) ** 2)
        freq_lo, freq_hi = ((10.0, 24.0) if family == "ood" else (2.0, 12.0))
        freq = torch.empty(1, device=device).uniform_(freq_lo, freq_hi)
        phase = torch.rand(1, device=device) * 2 * math.pi
        depth = 0.40 if family == "ood" else 0.25
        R[i] *= (1.0 - depth + depth * torch.cos(2 * math.pi * freq * wl + phase)).clamp_min(0.03)
    return R / R.amax(dim=-1, keepdim=True).clamp_min(1e-8)


def make_instrument_bank(wl, n_train=12, n_val=3, n_test=4, n_ood=4, channels=(12, 16, 24, 32), seed=7):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    bank = {"train": [], "val": [], "test": [], "ood": []}
    specs = [("train", n_train, "train"), ("val", n_val, "train"), ("test", n_test, "train"), ("ood", n_ood, "ood")]
    for split, count, fam in specs:
        for i in range(count):
            m = int(channels[(i + (1 if split == "ood" else 0)) % len(channels)])
            complexity = int(torch.randint(3, 8, (1,))) if fam == "train" else int(torch.randint(7, 11, (1,)))
            bank[split].append(random_response(m, wl, complexity=complexity, family=fam))
    torch.random.set_rng_state(state)
    return bank


def measure(R, x, noise_std=0.01):
    # R [M,N] or [B,M,N]
    if R.ndim == 2:
        y = torch.einsum("mn,bn->bm", R, x) / x.shape[-1]
    else:
        y = torch.einsum("bmn,bn->bm", R, x) / x.shape[-1]
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return y


def batch_for_instrument(R, x):
    return R.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous()


def ambiguous_pair_batch(R, wl, pairs=8, target_rel_distance=0.25, noise_std=0.0):
    """Construct positive spectra x+ and x- separated mainly in null(R).

    Uses d = (I - A^T(AA^T)^-1 A)v. The paired spectra share nearly the same
    noiseless detector measurement but are spectrally distinct. This creates a
    direct stress/training signal for non-identifiability.
    """
    N = wl.numel()
    A = R / N
    G = A @ A.T
    eye = torch.eye(G.shape[0], device=R.device, dtype=R.dtype)
    Ginv = torch.linalg.inv(G + 1e-8 * eye)
    xs = []
    for _ in range(pairs):
        base = 0.25 + 0.65 * random_spectra(1, wl, family="random")[0]
        v = torch.randn(N, device=R.device)
        d = v - A.T @ (Ginv @ (A @ v))
        d = d / torch.linalg.vector_norm(d).clamp_min(1e-8)
        safe = 0.85 * torch.min(base / d.abs().clamp_min(1e-8))
        desired = target_rel_distance * torch.linalg.vector_norm(base) / 2.0
        alpha = torch.minimum(safe, torch.as_tensor(desired, device=R.device))
        xp = (base + alpha * d).clamp_min(0.0)
        xm = (base - alpha * d).clamp_min(0.0)
        xs.extend([xp, xm])
    x = torch.stack(xs, dim=0)
    y = measure(R, x, noise_std=noise_std)
    # Make each pair share the same midpoint measurement to remove tiny numeric residuals.
    y2 = y.view(pairs, 2, -1).mean(1, keepdim=True).expand(-1, 2, -1).reshape(2 * pairs, -1)
    return x, y2
