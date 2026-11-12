"""Faithful, vectorised re-implementation of the ORIGINAL MusicID fingerprint (AudioModule + DBModule).

It reproduces every step of the original code, including its quirks, so the original
algorithm can be evaluated on thousands of songs (the original pure-Python version is
far too slow for that). tests/test_engine.py checks it against the original modules.

Quirks reproduced on purpose (see report):
  * mono = (L + R) / 2, and mono files are also halved
  * zero-padding to a power of two BEFORE filtering
  * 40 001-tap windowed-sinc low-pass filter, then 4-sample block averaging
  * windowed frames are stored as float16 before the FFT (numpy.insert into a float16 array)
  * peak threshold divides by 6 although only 5 bands are used        (fixed=True -> 5)
  * peak time stamp contains an extra j / 11025 s term (j = frequency bin) (fixed=True -> removed)
"""
import numpy as np
from scipy.signal import fftconvolve

SR = 44100
SR_DS = SR / 4                     # 11025.0, exactly as the original computes it
WIN = 1024
FREQ_RES = SR_DS / WIN             # 10.7666 Hz
FRAME_SEC = WIN / SR_DS            # 92.88 ms
BANDS = [(40, 80), (80, 160), (160, 250), (250, 350), (350, 512)]   # bands 3..7 of the original
_LPF_CACHE = {}


def original_lowpass(sr=SR, cutoff=5000, transition=0.0001):
    """AudioModule.LowPassFilter: windowed-sinc FIR with a Blackman window."""
    key = (sr, cutoff, transition)
    if key not in _LPF_CACHE:
        fc = cutoff / sr
        n_taps = int(np.ceil(4 / transition))
        if not n_taps % 2:
            n_taps += 1
        n = np.arange(n_taps)
        h = np.sinc(2 * fc * (n - (n_taps - 1) / 2.)) * np.blackman(n_taps)
        _LPF_CACHE[key] = h / np.sum(h)
    return _LPF_CACHE[key]


def to_mono_original(x):
    """StereoToMono: sum(axis=1) / 2 (also applied to mono input, which halves it)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x.sum(axis=1) / 2


def preprocess_original(mono, sr=SR):
    """Zero-pad to a power of two, low-pass at 5 kHz, average blocks of 4 -> 11025 Hz."""
    L = len(mono)
    N = 1 << (L - 1).bit_length()
    padded = np.zeros(N)
    padded[:L] = mono
    y = fftconvolve(padded, original_lowpass(sr))
    pad = (-len(y)) % 4                      # the original averages the last partial block too
    if pad:
        y = np.concatenate([y, np.zeros(pad)])
    return y.reshape(-1, 4).sum(axis=1) / 4


def hamming_original():
    n = np.arange(0, WIN)
    return 0.54 - 0.46 * np.cos((2 * np.pi * n) / WIN)


def spectrogram_original(d):
    """Non-overlapping Hamming frames (stored as float16, like the original) -> |FFT|, first 512 bins."""
    F = int(len(d) / WIN)
    frames = d[:F * WIN].reshape(F, WIN) * hamming_original()
    frames = frames.astype(np.float16).astype(np.float64)
    return np.abs(np.fft.fft(frames, axis=1))[:, :WIN // 2]


def peaks_original(S, t0=0.0, fixed=False):
    """LocatePowerfulFrequencies + time/frequency stamps.

    Returns times (s) and frequencies (Hz) of the kept peaks, frame-major then band order.
    fixed=True divides the mean by 5 (the number of bands actually used) and drops the
    j/11025 s term from the time stamp.
    """
    F = S.shape[0]
    if F == 0:
        return np.zeros(0), np.zeros(0)
    amps = np.empty((F, len(BANDS)))
    bins = np.empty((F, len(BANDS)), dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        seg = S[:, lo:hi]
        # MergeSort in the original sorts [amp, time, freq] descending -> on ties the larger bin wins
        rev = seg[:, ::-1]
        jj = (hi - lo - 1) - np.argmax(rev, axis=1)
        bins[:, b] = lo + jj
        amps[:, b] = seg[np.arange(F), jj]
    avg = amps.sum() / (F * (5 if fixed else 6))
    keep = amps > 0.3 * avg
    i_idx = np.repeat(np.arange(F), len(BANDS)).reshape(F, len(BANDS))
    i_k, j_k = i_idx[keep], bins[keep]
    if fixed:
        t = (i_k * WIN) / SR_DS + t0
    else:
        t = (i_k * WIN) / SR_DS + ((WIN / SR_DS) / WIN) * j_k + t0
    f = j_k * FREQ_RES
    return t.astype(np.float64), f.astype(np.float64)


def fingerprint_original(audio, sr=SR, fixed=False, t0=0.0):
    """Audio (int16/float, mono or (n, ch)) at 44.1 kHz -> constellation map (t, f)."""
    if sr != SR:
        raise ValueError('v1 expects 44.1 kHz input (resample first, as the app does)')
    d = preprocess_original(to_mono_original(audio), sr)
    return peaks_original(spectrogram_original(d), t0=t0, fixed=fixed)


# ---------------- hashing (DBModule.GenerateAddressCouple*) ----------------
def hashes_original(t, f):
    """Anchor = point i-3, target zone = points i..i+4 (list order). Returns packed keys and anchor times (ms).

    address = (round(f_a/10), round(f_p/10), round(1000*(t_p - t_a)));  couple time = round(1000*t_a).
    The original concatenates binary strings; here the same triple is packed into one int64.
    """
    n = len(t)
    if n < 9:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    i = np.arange(3, n - 5)
    a = i - 3
    p = i[:, None] + np.arange(5)[None, :]
    A = np.rint(f[a] / 10).astype(np.int64)[:, None].repeat(5, 1)
    B = np.rint(f[p] / 10).astype(np.int64)
    DT = np.rint((t[p] - t[a][:, None]) * 1000).astype(np.int64)
    keys = (A << 34) | (B << 20) | DT
    ta = np.rint(t[a] * 1000).astype(np.int64)[:, None].repeat(5, 1)
    return keys.ravel(), ta.ravel()
