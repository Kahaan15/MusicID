"""Unit tests. Run from the project folder:  python -m pytest -q tests"""
import os, sys, wave
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

import AudioModule, DBModule
from research import fingerprint_v1 as v1, fingerprint_v2 as v2, stats
from research.index import HashIndex, offset_votes, match_original_rules


def synthetic_song(seconds, sr=44100, seed=0):
    """A 'song' of random chords + noise, stereo int16."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for start in np.arange(0, seconds, 0.25):
        m = (t >= start) & (t < start + 0.25)
        for f in rng.uniform(300, 4800, 4):
            x[m] += np.sin(2 * np.pi * f * t[m]) * rng.uniform(0.2, 1)
    x += 0.05 * rng.standard_normal(n)
    x = x / np.abs(x).max() * 20000
    return np.stack([x, x * 0.9], 1).astype(np.int16)


def write_wav(path, x, sr=44100):
    with wave.open(path, 'wb') as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x.tobytes())


def test_handwritten_fft_matches_numpy():
    rng = np.random.default_rng(1)
    for N in (8, 64, 1024):
        x = rng.standard_normal(N)
        ours = np.array(AudioModule.FastFourierTransform(list(x), N))
        assert np.allclose(ours, np.fft.fft(x), atol=1e-9 * N)


def test_v1_replica_is_bit_identical_to_original(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = synthetic_song(6.0)
    write_wav('clip.wav', x)
    AudioModule.GenerateConstellationMap('clip.wav', 1, 0)
    ref = np.loadtxt('SampleFingerprint.txt')
    n = (len(x) // 2048) * 2048
    t, f = v1.fingerprint_original(x[:n])
    assert np.array_equal(ref[:, 0], t) and np.array_equal(ref[:, 1], f)
    # hashes and couples equal the original DBModule output
    addr, cpl = DBModule.GenerateAddressCoupleQUERY(os.path.join(os.getcwd(), 'SampleFingerprint.txt'))
    k, ta = v1.hashes_original(t, f)
    orig = [(int(a[:9], 2), int(a[9:18], 2), int(a[18:], 2)) for row in addr for a in row]
    ours = [(kk >> 34, (kk >> 20) & 0x3FFF, kk & 0xFFFFF) for kk in k.tolist()]
    assert orig == ours
    assert [int(c, 2) for row in cpl for c in row] == ta.tolist()


def test_index_lookup_matches_bruteforce():
    rng = np.random.default_rng(2)
    per_song = [(rng.integers(0, 50, 200), rng.integers(0, 1000, 200)) for _ in range(5)]
    idx = HashIndex.build(per_song, [1000] * 5)
    qk, qt = rng.integers(0, 50, 30), rng.integers(0, 100, 30)
    s, d, qi = idx.lookup(qk, qt)
    got = sorted(zip(s.tolist(), d.tolist(), qi.tolist()))
    want = sorted((sid, int(tt - qt[i]), i) for i in range(30) for sid, (k, t) in enumerate(per_song)
                  for kk, tt in zip(k, t) if kk == qk[i])
    assert got == want


def test_v1_original_rules_find_the_song():
    song = synthetic_song(20.0, seed=3)
    t, f = v1.fingerprint_original(song)
    k, ta = v1.hashes_original(t, f)
    other = synthetic_song(20.0, seed=4)
    t2, f2 = v1.fingerprint_original(other)
    k2, ta2 = v1.hashes_original(t2, f2)
    idx = HashIndex.build([(k2, ta2), (k, ta)], [1, 1])
    s0 = 5 * 2048 * 20
    q = song[s0:s0 + 2048 * 100]
    qt, qf = v1.fingerprint_original(q)
    qk, qa = v1.hashes_original(qt, qf)
    sid, score, _ = match_original_rules(idx, qk, qa, 4)
    assert sid == 1 and score > 39


def test_v2_finds_shifted_clip_and_offset():
    song = synthetic_song(30.0, seed=5).astype(np.float64)
    k, ta, aid, nf = v2.fingerprint(song, 44100)
    idx = HashIndex.build([(k, ta)], [nf])
    start = 12.3
    q = song[int(start * 44100): int((start + 4) * 44100)]
    qk, qt, qa, qn = v2.fingerprint(q, 44100)
    r = offset_votes(idx, qk, qt, qanchor=qa)
    assert r['song'][0] == 0
    assert abs(r['best_delta'][0] * v2.FRAME_SEC - start) < 0.1


def test_offset_merge_joins_split_votes():
    # votes at delta 10 and 11 for song 0 should merge into one window of 2 + 3
    idx = HashIndex.build([(np.array([1, 1, 2, 2, 2]), np.array([10, 10, 11, 11, 11]))], [100])
    r = offset_votes(idx, np.array([1, 2]), np.array([0, 0]), radius=1)
    assert r['score'][0] == 5


def test_poisson_pvalue_is_valid_under_null():
    """Random votes (true H0): the false-match rate at alpha must not exceed alpha (plus noise)."""
    rng = np.random.default_rng(6)
    n_songs, K_song, K_query, trials, alpha = 200, 600, 60, 400, 0.05
    false = 0
    for _ in range(trials):
        M = rng.poisson(20, n_songs)
        scores = []
        for m in M:
            d = rng.integers(0, K_song + K_query, m)
            c = np.bincount(d, minlength=K_song + K_query + 2)
            merged = c + np.roll(c, 1) + np.roll(c, -1)
            scores.append(merged.max() if m else 0)
        votes = {'song': np.arange(n_songs), 'score': np.array(scores), 'M': M}
        p, *_ = stats.family_pvalue(votes, np.full(n_songs, K_song), K_query)
        false += p < alpha
    assert false / trials <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / trials)


def test_conformal_threshold_controls_fpr():
    rng = np.random.default_rng(7)
    alpha, fails = 0.1, []
    for _ in range(300):
        cal, test = rng.standard_normal(200), rng.standard_normal(1)
        fails.append(test[0] > stats.conformal_threshold(cal, alpha))
    assert np.mean(fails) <= alpha + 0.05


def test_sequential_bonferroni_is_stricter_than_naive():
    p = [0.03, 0.2, 0.004]
    assert stats.sequential_decision(p, 0.05, 'naive') == (True, 0)
    assert stats.sequential_decision(p, 0.05, 'bonferroni') == (True, 2)


def test_app_engine_add_identify_reject(tmp_path, monkeypatch):
    import EngineModule
    monkeypatch.setattr(EngineModule, 'INDEX_PATH', str(tmp_path / 'index.npz'))
    monkeypatch.setattr(EngineModule, 'SONGS_PATH', str(tmp_path / 'songs.json'))
    monkeypatch.setattr(EngineModule, 'LIB_DIR', str(tmp_path))
    eng = EngineModule.Engine()
    songs = [synthetic_song(30.0, seed=s).astype(np.float64) for s in (10, 11, 12)]
    for n, x in enumerate(songs):
        eng.add_song(x, 44100, {'title': f'song {n}', 'artist': 'test'})
    eng.save()
    eng = EngineModule.Engine()                       # reload from disk
    q = songs[1][int(8.0 * 44100): int(13.0 * 44100)]
    r = eng.identify(q, 44100)
    assert r['matched'] and r['song']['title'] == 'song 1' and abs(r['offset_sec'] - 8.0) < 0.15
    stranger = synthetic_song(5.0, seed=99).astype(np.float64)
    assert not eng.identify(stranger, 44100)['matched']
