"""Neural audio fingerprint learned with contrastive learning (in the spirit of Chang et al., ICASSP 2021).

Each 1-second segment is mapped to a 128-d unit vector. Training pairs are (clean segment,
the same segment degraded: shifted in time, noise, room echo, EQ, small speed change), and
the NT-Xent loss pulls each pair together while pushing apart the other 2B-2 segments of
the batch. Songs used for training are disjoint from the evaluation library.

Search: database = embeddings of every library track at 0.5 s hops. Each query segment
finds its nearest database segments; votes are aggregated per (track, time offset), the
same offset-consistency idea as the classical method.
"""
import math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 11025
SEG = SR                      # 1 s segments
HOP_DB = SR // 2              # database hop 0.5 s
N_FFT, HOP, N_MELS = 1024, 256, 64
DIM = 128


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=150., fmax=5000.):
    def hz2mel(h): return 2595 * np.log10(1 + h / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        l, c, r = hz[m], hz[m + 1], hz[m + 2]
        fb[m] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return torch.tensor(fb, dtype=torch.float32)


class Frontend(nn.Module):
    """Waveform (B, n) -> normalised log-mel (B, 1, 64, frames)."""
    def __init__(self):
        super().__init__()
        self.register_buffer('fb', mel_filterbank())
        self.register_buffer('win', torch.hann_window(N_FFT))

    def forward(self, x):
        X = torch.stft(x, N_FFT, HOP, window=self.win, center=False, return_complex=True)
        m = torch.log(self.fb @ (X.abs() ** 2) + 1e-6)
        m = m - m.mean(dim=(1, 2), keepdim=True)              # gain invariance
        return m.unsqueeze(1)


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FingerprintNet(nn.Module):
    """Small CNN (~0.5 M parameters, ~4 M multiply-adds per segment) so it trains on a laptop CPU.

    64x40 log-mel -> 32x20 -> 16x10 -> 8x5 -> 4x3 feature maps, then the flattened map
    (keeps where in time/frequency things happen) -> 128-d unit vector.
    """
    def __init__(self):
        super().__init__()
        self.front = Frontend()
        self.body = nn.Sequential(block(1, 16, 2), block(16, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, 128, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 3, 256), nn.ReLU(inplace=True), nn.Linear(256, DIM))

    def forward(self, wave):
        return F.normalize(self.head(self.body(self.front(wave))), dim=1)


# ---------------- augmentation (batched, torch) ----------------
def _pink(B, n, g):
    X = torch.fft.rfft(torch.randn(B, n, generator=g), dim=1)
    f = torch.arange(X.shape[1], dtype=torch.float32).clamp(min=1)
    y = torch.fft.irfft(X / f.sqrt(), n, dim=1)
    return y / (y.std(dim=1, keepdim=True) + 1e-8)


def _mix(x, noise, snr_db):
    ps = x.pow(2).mean(1, keepdim=True) + 1e-8
    pn = noise.pow(2).mean(1, keepdim=True) + 1e-8
    return x + noise * torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))


_BANK = {}


def _banks(g, n=SEG):
    """Pre-computed banks of pink noise and room impulse responses (RT60 0.2-0.8 s), reused every step."""
    if not _BANK:
        _BANK['pink'] = _pink(2048, n, g)
        L = int(0.4 * SR) * 2
        nfft = 1 << (n + L).bit_length()
        rt = torch.empty(512, 1).uniform_(0.2, 0.8, generator=g)
        t = torch.arange(L).float()[None] / SR
        ir = torch.randn(512, L, generator=g) * torch.exp(-6.908 * t / rt) * 0.3
        ir[:, :int(0.003 * SR)] = 0
        ir[:, 0] = 1
        _BANK['irf'], _BANK['nfft'] = torch.fft.rfft(ir, nfft), nfft
    return _BANK


def _reverb(x, g):
    B, n = x.shape
    bank = _banks(g, n)
    k = torch.randint(0, bank['irf'].shape[0], (B,), generator=g)
    nfft = bank['nfft']
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * bank['irf'][k], nfft)[:, :n]
    return y * (x.std(1, keepdim=True) / (y.std(1, keepdim=True) + 1e-8))


def _eq(x, g):
    B, n = x.shape
    X = torch.fft.rfft(x, dim=1)
    f = torch.fft.rfftfreq(n, 1 / SR)[None]
    lo = torch.empty(B, 1).uniform_(80, 400, generator=g)
    hi = torch.empty(B, 1).uniform_(3000, 5500, generator=g)
    mask = torch.sigmoid((f - lo) / 30) * torch.sigmoid((hi - f) / 200)
    return torch.fft.irfft(X * mask, n, dim=1)


def augment(clean_long, offsets, bg, g):
    """clean_long: (B, SEG + 2*pad) source audio; offsets: sample offset of the positive view.

    Returns the degraded positive views (B, SEG).
    """
    B = clean_long.shape[0]
    # small speed change (0.97..1.03) applied by resampling a slightly longer/shorter window
    speeds = torch.where(torch.rand(B, generator=g) < 0.3, torch.empty(B).uniform_(0.97, 1.03, generator=g), torch.ones(B))
    # vectorised linear-interpolation resampling: output sample k reads input position offset + k*speed
    pos = offsets[:, None].float() + torch.arange(SEG).float()[None] * speeds[:, None]
    i0 = pos.floor().long().clamp(max=clean_long.shape[1] - 2)
    frac = pos - i0.float()
    x = torch.gather(clean_long, 1, i0) * (1 - frac) + torch.gather(clean_long, 1, i0 + 1) * frac
    r = torch.rand(B, 4, generator=g)
    m = r[:, 0] < 0.5
    if m.any():
        x = x.clone(); x[m] = _reverb(x[m], g)
    m = r[:, 1] < 0.5
    if m.any():
        x = x.clone(); x[m] = _eq(x[m], g)
    snr = torch.empty(B, 1).uniform_(0, 20, generator=g)
    pink = _banks(g)['pink'][torch.randint(0, 2048, (B,), generator=g)]
    x = torch.where((r[:, 2] < 0.8)[:, None], _mix(x, pink, snr), x)
    snr2 = torch.empty(B, 1).uniform_(5, 20, generator=g)
    x = torch.where((r[:, 3] < 0.3)[:, None], _mix(x, bg, snr2), x)
    return x * torch.empty(B, 1).uniform_(0.3, 1.5, generator=g)


def nt_xent(z1, z2, tau=0.05):
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    n = z1.shape[0]
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def train(audio_mm, track_ids, steps=3000, batch=256, lr=1e-3, seed=0, log=print, out=None, threads=12):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FingerprintNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    pad = int(0.25 * SR) + int(0.05 * SEG)
    n_len = audio_mm.shape[1]
    t0 = time.time()
    hist = []
    import threading, queue as _q
    batches = _q.Queue(maxsize=4)

    def producer():                      # builds augmented batches while the model trains
        for _ in range(steps):
            tr = rng.choice(track_ids, batch)
            st = rng.integers(pad, n_len - SEG - 2 * pad, batch)
            long = np.stack([audio_mm[t, s - pad: s + SEG + pad] for t, s in zip(tr, st)]).astype(np.float32) / 32768
            bgt = rng.choice(track_ids, batch)
            bgs = rng.integers(0, n_len - SEG, batch)
            bg = np.stack([audio_mm[t, s: s + SEG] for t, s in zip(bgt, bgs)]).astype(np.float32) / 32768
            long, bg = torch.from_numpy(long), torch.from_numpy(bg)
            offs = torch.from_numpy(rng.integers(0, 2 * pad - int(0.05 * SEG), batch))     # +/- 0.25 s shift
            batches.put((long[:, pad: pad + SEG].clone(), augment(long, offs, bg, g)))

    threading.Thread(target=producer, daemon=True).start()
    for step in range(steps):
        anchor, pos = batches.get()
        model.train()
        loss = nt_xent(model(anchor), model(pos))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if step % 50 == 0 or step == steps - 1:
            log(f'step {step} loss {np.mean(hist[-50:]):.3f} {time.time() - t0:.0f}s')
        if out and (step % 500 == 0 or step == steps - 1):
            torch.save(model.state_dict(), out)
    return model, hist


@torch.no_grad()
def embed_segments(model, x, hop=HOP_DB, batch=512):
    """Mono 11 kHz signal -> (n_segments, 128) embeddings of 1-s windows at `hop`."""
    model.eval()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if len(x) < SEG:
        x = F.pad(x, (0, SEG - len(x)))
    n = 1 + (len(x) - SEG) // hop
    segs = x.unfold(0, SEG, hop)[:n]
    out = [model(segs[i:i + batch]) for i in range(0, n, batch)]
    return torch.cat(out)


class NeuralIndex:
    def __init__(self, emb, track, seg):
        self.emb, self.track, self.seg = emb, track, seg       # (N,128) float32, track id, segment index

    def subset(self, n):
        m = self.track < n
        return NeuralIndex(self.emb[m], self.track[m], self.seg[m])

    @torch.no_grad()
    def search(self, q, k=20):
        """q: (n_q, 128) query segment embeddings (hop 0.5 s). Returns (track, score, offset, second score).

        score = mean over query segments of the best similarity that agrees with the offset.
        """
        sim = q @ self.emb.t()
        val, idx = sim.topk(k, dim=1)
        tr = self.track[idx]                                  # (n_q, k)
        off = self.seg[idx] - torch.arange(q.shape[0])[:, None]
        key = tr.long() * 100000 + (off.long() + 1000)
        key, val, qi = key.flatten(), val.flatten(), torch.arange(q.shape[0]).repeat_interleave(k)
        # one vote per query segment per (track, offset): keep the best similarity
        uniq, inv = torch.unique(torch.stack([key, qi]), dim=1, return_inverse=True)
        best = torch.full((uniq.shape[1],), -1.0).scatter_reduce(0, inv, val, 'amax')
        ukey = uniq[0]
        keys2, inv2 = torch.unique(ukey, return_inverse=True)
        score = torch.zeros(len(keys2)).scatter_add(0, inv2, best) / q.shape[0]
        order = torch.argsort(score, descending=True)
        top = order[0]
        tk = keys2[top].item()
        top_track = tk // 100000
        second = 0.0
        for o in order[1:]:
            if keys2[o].item() // 100000 != top_track:
                second = score[o].item()
                break
        return int(top_track), float(score[top]), int(tk % 100000 - 1000), second
