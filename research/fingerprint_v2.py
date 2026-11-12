"""Improved classical fingerprint (v2), following Wang (2003) more closely.

Changes vs the original (each one is measured in the ablation):
  * proper anti-aliasing: polyphase resampling to 11025 Hz (no 40 001-tap filter, any input rate)
  * 50 % overlapping Hann frames (hop 512 = 46.4 ms) -> robust to frame misalignment
  * peaks = local maxima of the log spectrogram in a time-frequency neighbourhood,
    with density control (at most PEAKS_PER_SEC strongest per second)
  * real target zones: each anchor is paired with up to FAN_OUT later peaks within
    1..MAX_DT frames and +/- MAX_DF bins (instead of "the next 5 points in a list")
  * time in integer frames, so offsets have no rounding split
  * 24-bit hash = f_anchor (9 bits) | f_target (9 bits) | dt (6 bits)
"""
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from scipy.ndimage import maximum_filter

SR = 11025
WIN = 1024
HOP = 512
FRAME_SEC = HOP / SR                 # 46.4 ms per time unit
F_LO, F_HI = 20, 500                 # bins (215 Hz .. 5.4 kHz)
NEIGH_T, NEIGH_F = 7, 21             # local-maximum neighbourhood (frames, bins)
PEAKS_PER_SEC = 30
FAN_OUT = 6
MAX_DT = 40                          # frames (~1.9 s)
MAX_DF = 100                         # bins (~1 kHz)
_WINDOW = np.hanning(WIN + 1)[:-1]


def to_11k(audio, sr):
    """Mono (mean of channels) float signal resampled to 11025 Hz with a polyphase anti-alias filter."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x


def log_spectrogram(x):
    n = 1 + max(0, (len(x) - WIN) // HOP)
    if len(x) < WIN:
        x = np.concatenate([x, np.zeros(WIN - len(x))])
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * _WINDOW
    S = np.abs(np.fft.rfft(frames, axis=1))[:, :WIN // 2]
    return np.log(S + 1e-3 * (S.max() + 1e-12) + 1e-9)


def peaks(S):
    """(frame, bin) of local maxima, strongest PEAKS_PER_SEC per second, frequency band F_LO..F_HI."""
    band = S[:, F_LO:F_HI]
    mx = maximum_filter(band, size=(NEIGH_T, NEIGH_F), mode='constant', cval=-np.inf)
    cand = (band == mx) & (band > np.median(band) + 1.0)
    t, f = np.nonzero(cand)
    if len(t) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    amp = band[t, f]
    sec = int(round(1 / FRAME_SEC))
    block = t // sec
    order = np.lexsort((-amp, block))
    t, f, block = t[order], f[order], block[order]
    rank = np.arange(len(t)) - np.searchsorted(block, block, 'left')
    keep = rank < PEAKS_PER_SEC
    t, f = t[keep], f[keep] + F_LO
    o = np.lexsort((f, t))
    return t[o].astype(np.int64), f[o].astype(np.int64)


def hashes(t, f, fan_out=FAN_OUT):
    """Anchor/target-zone pairs -> (24-bit keys, anchor frame, anchor index)."""
    n = len(t)
    if n < 2:
        z = np.zeros(0, np.int64)
        return z, z, z
    keys, ta, aid = [], [], []
    # look ahead over the next few hundred peaks at most (sorted by time)
    for k in range(1, 200):
        if k >= n:
            break
        a = np.arange(n - k)
        b = a + k
        dt = t[b] - t[a]
        ok = (dt >= 1) & (dt <= MAX_DT) & (np.abs(f[b] - f[a]) <= MAX_DF)
        if not ok.any() and (dt > MAX_DT).all():
            break
        keys.append(np.where(ok, (f[a] << 15) | (f[b] << 6) | dt, -1))
        ta.append(t[a]); aid.append(a)
    K = np.stack([np.concatenate([kk, np.full(n - len(kk), -1)]) for kk in keys], 1)     # n x lookahead
    valid = K >= 0
    rank = np.cumsum(valid, axis=1)
    sel = valid & (rank <= fan_out)
    rows, cols = np.nonzero(sel)
    return K[rows, cols].astype(np.int64), t[rows], rows.astype(np.int64)


def fingerprint(audio, sr):
    """Audio at any rate -> (keys, anchor frames, anchor ids, number of frames)."""
    x = to_11k(audio, sr)
    S = log_spectrogram(x)
    t, f = peaks(S)
    k, ta, aid = hashes(t, f)
    return k, ta, aid, S.shape[0]


def fingerprint_11k(x):
    """Same, for a signal that is already mono at 11025 Hz."""
    S = log_spectrogram(np.asarray(x, dtype=np.float64))
    t, f = peaks(S)
    k, ta, aid = hashes(t, f)
    return k, ta, aid, S.shape[0]
