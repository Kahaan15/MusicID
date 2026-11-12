"""Query degradations used in the benchmark (all seeded, all applied to mono float audio)."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt, resample


def pink_noise(n, rng):
    """1/f noise made by shaping white noise in the frequency domain."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    y = np.fft.irfft(X / np.sqrt(f), n)
    return y / (np.std(y) + 1e-12)


def add_noise(x, snr_db, rng, kind='pink'):
    noise = pink_noise(len(x), rng) if kind == 'pink' else rng.standard_normal(len(x))
    p_sig = np.mean(x ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + noise * np.sqrt(p_noise)


def room_ir(sr, rt60, rng):
    """Synthetic room impulse response: direct path + exponentially decaying noise tail."""
    n = int(sr * rt60)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.908 * t / rt60) * 0.3
    tail[: int(0.003 * sr)] = 0               # 3 ms before the first reflections
    ir = tail
    ir[0] = 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


def reverb(x, sr, rt60, rng):
    y = fftconvolve(x, room_ir(sr, rt60, rng))[: len(x)]
    return y * (np.std(x) / (np.std(y) + 1e-12))


def bandpass(x, sr, lo=150, hi=5000):
    sos = butter(4, [lo, hi], btype='band', fs=sr, output='sos')
    return sosfilt(sos, x)


def speed(x, factor):
    """Play back `factor` times faster (changes tempo AND pitch, like a slightly wrong turntable)."""
    return resample(x, int(round(len(x) / factor)))


def phone_mic(x, sr, rng):
    """Realistic phone-in-a-room recording: band-limited mic, room echo, background noise at 10 dB."""
    y = reverb(bandpass(x, sr), sr, 0.4, rng)
    return add_noise(y, 10, rng)


CONDITIONS = ['clean', 'pink20', 'pink10', 'pink5', 'pink0', 'reverb', 'mic', 'speed+1%', 'speed-2%', 'speed+5%']


def apply(x, sr, cond, rng):
    if cond == 'clean':
        return x.copy()
    if cond.startswith('pink'):
        return add_noise(x, float(cond[4:]), rng)
    if cond == 'reverb':
        return reverb(x, sr, 0.6, rng)
    if cond == 'mic':
        return phone_mic(x, sr, rng)
    if cond.startswith('speed'):
        return speed(x, 1 + float(cond[5:-1]) / 100)
    raise ValueError(cond)
