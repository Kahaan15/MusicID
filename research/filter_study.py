"""Does the 40 001-tap anti-aliasing filter matter?

Three ways to go from 44.1 kHz to 11.025 kHz, everything else identical (v1 fixed fingerprint):
  orig   : 40 001-tap windowed sinc (5 kHz, 4.4 Hz transition) + average of 4   (the app)
  kaiser : Kaiser-window FIR designed for the real requirement + average of 4
           (pass 5000 Hz, stop 6025 Hz = 11025-5000, where aliases would land on bins we use; 60 dB)
  poly   : scipy resample_poly(1/4) (polyphase decimation with its own anti-alias filter)

Measured: filter length, time per 29 s song, peak agreement with `orig`, and identification
(true/false match rates) on a 1000-song library.
python -m research.filter_study
"""
import os, time, json, pickle
import numpy as np
from scipy.signal import fftconvolve, kaiserord, firwin, resample_poly, freqz
from concurrent.futures import ProcessPoolExecutor

from research import data, fingerprint_v1 as v1, stats
from research.index import HashIndex, offset_votes
from research.bench import names, query_audio, WORK, RESULTS, CLIP, V1_UNIT, V1_SONG_FRAMES

N_SONGS = 1000


def kaiser_taps():
    numtaps, beta = kaiserord(60, (6025 - 5000) / (44100 / 2))
    numtaps |= 1
    return firwin(numtaps, (5000 + 6025) / 2, window=('kaiser', beta), fs=44100)


def preprocess(mono, method):
    L = len(mono)
    N = 1 << (L - 1).bit_length()
    padded = np.zeros(N); padded[:L] = mono
    if method == 'poly':
        return resample_poly(padded, 1, 4)
    h = v1.original_lowpass() if method == 'orig' else kaiser_taps()
    y = fftconvolve(padded, h)
    if method == 'kaiser':
        y = y[(len(h) - 1) // 2:]                    # remove the group delay of the short filter
    y = np.concatenate([y, np.zeros((-len(y)) % 4)])
    return y.reshape(-1, 4).sum(1) / 4


def fp(mono, method):
    S = v1.spectrogram_original(preprocess(mono, method))
    t, f = v1.peaks_original(S, fixed=True)
    return t, f


def _song(name):
    x = data.load44(name)[: int(CLIP * 44100)]
    x = x[: (len(x) // 2048) * 2048] * 32767.0
    mono = x.mean(axis=1)
    out = {}
    for m in ('orig', 'kaiser', 'poly'):
        t0 = time.perf_counter()
        t, f = fp(mono, m)
        dt = time.perf_counter() - t0
        k, ta = v1.hashes_original(t, f)
        out[m] = (k, ta.astype(np.int32), dt, t, f)
    return out


def _query(spec):
    y = query_audio(spec)
    y = y[: (len(y) // 2048) * 2048]
    out = {}
    for m in ('orig', 'kaiser', 'poly'):
        t, f = fp(y, m)
        k, ta = v1.hashes_original(t, f)
        out[m] = (k, ta, np.repeat(np.arange(len(k) // 5), 5))
    return spec, out


def main(workers=8):
    tracks, sp = names()
    lib = [tracks[i] for i in sp['library'][:N_SONGS]]
    with ProcessPoolExecutor(workers) as ex:
        songs = list(ex.map(_song, lib, chunksize=4))
    res = {'taps': {'orig': len(v1.original_lowpass()), 'kaiser': len(kaiser_taps()), 'poly': 'scipy default'}}
    res['sec_per_song'] = {m: float(np.mean([s[m][2] for s in songs])) for m in ('orig', 'kaiser', 'poly')}
    agree = {}
    for m in ('kaiser', 'poly'):
        j = []
        for s in songs:
            a = set(zip(np.round(s['orig'][3], 3), np.round(s['orig'][4], 1)))
            b = set(zip(np.round(s[m][3], 3), np.round(s[m][4], 1)))
            j.append(len(a & b) / max(1, len(a | b)))
        agree[m] = float(np.mean(j))
    res['peak_jaccard_vs_orig'] = agree
    idx = {m: HashIndex.build([(s[m][0], s[m][1]) for s in songs], [V1_SONG_FRAMES] * len(songs)) for m in ('orig', 'kaiser', 'poly')}
    # queries
    rng = np.random.default_rng(7)
    neg = sp['negative'][500:]
    specs = []
    for cond in ('clean', 'pink10', 'mic'):
        for _ in range(300):
            s = int(rng.integers(N_SONGS))
            specs.append({'grid': 'filter', 'cond': cond, 'L': 5.0, 'label': 1, 'song': s, 'track': lib[s],
                          'start': float(rng.uniform(0, CLIP - 5.2)), 'seed': int(rng.integers(2**31))})
        for _ in range(300):
            specs.append({'grid': 'filter', 'cond': cond, 'L': 5.0, 'label': 0, 'song': -1, 'track': tracks[int(rng.choice(neg))],
                          'start': float(rng.uniform(0, CLIP - 5.2)), 'seed': int(rng.integers(2**31))})
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for spec, q in ex.map(_query, specs, chunksize=8):
            for m in ('orig', 'kaiser', 'poly'):
                k, t, a = q[m]
                votes = offset_votes(idx[m], k, t, qanchor=a, unit=V1_UNIT)
                ks = idx[m].song_frames[votes['song']] if len(votes['song']) else np.zeros(0)
                p, top, c, _ = stats.family_pvalue(votes, ks, 5.0 / v1.FRAME_SEC)
                rows.append({'cond': spec['cond'], 'label': spec['label'], 'song': spec['song'], 'method': m, 'p': p, 'top': top})
    table = {}
    for m in ('orig', 'kaiser', 'poly'):
        for cond in ('clean', 'pink10', 'mic'):
            r = [x for x in rows if x['method'] == m and x['cond'] == cond]
            pos = [x for x in r if x['label'] == 1]
            neg_ = [x for x in r if x['label'] == 0]
            table[f'{m}/{cond}'] = {'tpr': float(np.mean([(x['p'] < 0.01) and x['top'] == x['song'] for x in pos])),
                                    'fpr': float(np.mean([x['p'] < 0.01 for x in neg_]))}
    res['identification'] = table
    # frequency responses for the figure
    w, h_orig = freqz(v1.original_lowpass(), worN=8192, fs=44100)
    _, h_k = freqz(kaiser_taps(), worN=8192, fs=44100)
    _, h_avg = freqz(np.ones(4) / 4, worN=8192, fs=44100)
    res['freqz'] = {'f': w.tolist(), 'orig': (20 * np.log10(np.abs(h_orig) + 1e-12)).tolist(),
                    'kaiser': (20 * np.log10(np.abs(h_k) + 1e-12)).tolist(), 'avg4': (20 * np.log10(np.abs(h_avg) + 1e-12)).tolist()}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, 'filter_study.json'), 'w'))
    print(json.dumps({k: v for k, v in res.items() if k != 'freqz'}, indent=1))


if __name__ == '__main__':
    main()
