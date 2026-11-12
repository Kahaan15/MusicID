"""Benchmark pipeline: build the indexes, generate degraded queries, run every method.

Usage (from the project folder):
    python -m research.bench build          # v1 (original + fixed) and v2 indexes over the 4000-song library
    python -m research.bench prepare        # generate all query sets, fingerprint them (parallel)
    python -m research.bench classical      # match prepared queries with the classical engines
    python -m research.bench neural         # embed library + queries with the trained neural model and search
Results are written to research/results/*.pkl and summarised by research/analyze.py.
"""
import os, sys, time, pickle, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from research import data, augment, fingerprint_v1 as v1, fingerprint_v2 as v2, stats
from research.index import HashIndex, offset_votes, match_original_rules

RESULTS = os.path.join(data.ROOT, 'research', 'results')
WORK = os.path.join(data.DATA, 'work')
CLIP = 29.0
V1_UNIT = v1.FRAME_SEC * 1000                  # v1 stores ms; offsets are compared in frames (92.9 ms)
V1_SONG_FRAMES = int(CLIP / v1.FRAME_SEC)
V2_SONG_FRAMES = (int(CLIP * v2.SR) - v2.WIN) // v2.HOP + 1
N_LIB = 4000
SEED = 2026
PHIS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def names():
    tracks, _ = data.cached()
    sp = data.splits()
    return tracks, sp


# ------------------------------------------------------------------ indexes
def _v1_song(name):
    x = data.load44(name)[: int(CLIP * 44100)]
    x = x[: (len(x) // 2048) * 2048] * 32767.0          # same scale as the 16-bit WAVs the app reads
    S = v1.spectrogram_original(v1.preprocess_original(v1.to_mono_original(x)))
    out = {}
    for fixed in (False, True):
        t, f = v1.peaks_original(S, fixed=fixed)
        k, ta = v1.hashes_original(t, f)
        out[fixed] = (k, ta.astype(np.int32))
    return out


def _v2_song(i):
    _, mm = data.cached()
    k, ta, aid, nf = v2.fingerprint_11k(mm[i].astype(np.float64) / 32768)
    return k, ta.astype(np.int32)


def build(workers=10):
    os.makedirs(WORK, exist_ok=True)
    tracks, sp = names()
    lib = [tracks[i] for i in sp['library'][:N_LIB]]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_v1_song, lib, chunksize=4))
    for fixed, tag in ((False, 'v1o'), (True, 'v1f')):
        idx = HashIndex.build([r[fixed] for r in res], [V1_SONG_FRAMES] * len(res))
        idx.save(os.path.join(WORK, f'index_{tag}.npz'))
        print(tag, 'hashes', len(idx.keys), 'MB', idx.nbytes // 2**20, flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res2 = list(ex.map(_v2_song, list(sp['library'][:N_LIB]), chunksize=16))
    idx = HashIndex.build(res2, [V2_SONG_FRAMES] * len(res2))
    idx.save(os.path.join(WORK, 'index_v2.npz'))
    print('v2 hashes', len(idx.keys), 'MB', idx.nbytes // 2**20, flush=True)
    json.dump({'v1_sec': t1 - t0, 'v2_sec': time.time() - t1, 'n': len(lib)},
              open(os.path.join(WORK, 'build_time.json'), 'w'))


# ------------------------------------------------------------------ queries
def make_specs():
    """All query sets. Positives come from library songs, negatives from songs never indexed."""
    tracks, sp = names()
    lib, neg = sp['library'][:N_LIB], sp['negative']
    neg_cal, neg_test = neg[:500], neg[500:]
    rng = np.random.default_rng(SEED)
    specs = []

    def add(grid, cond, L, label, song=None, track=None):
        fac = 1 + float(cond[5:-1]) / 100 if cond.startswith('speed') else 1.0
        start = float(rng.uniform(0, CLIP - L * fac - 0.2))
        specs.append({'id': len(specs), 'grid': grid, 'cond': cond, 'L': L, 'label': label,
                      'song': -1 if song is None else int(song), 'track': tracks[int(track)],
                      'start': start, 'seed': int(rng.integers(2**31))})

    # calibration / ML-training set: random condition and length
    for _ in range(2000):
        s = rng.integers(N_LIB)
        add('calib', str(rng.choice(augment.CONDITIONS)), float(rng.choice([1, 2, 3, 5, 10])), 1, s, lib[s])
    for _ in range(1000):
        add('calib', str(rng.choice(augment.CONDITIONS)), float(rng.choice([1, 2, 3, 5, 10])), 0, None, rng.choice(neg_cal))
    # main robustness grid (5 s queries)
    for cond in augment.CONDITIONS:
        for _ in range(400):
            s = rng.integers(N_LIB); add('main', cond, 5.0, 1, s, lib[s])
        for _ in range(400):
            add('main', cond, 5.0, 0, None, rng.choice(neg_test))
    # query length grid
    for cond in ['clean', 'pink5', 'mic']:
        for L in [1.0, 2.0, 3.0, 10.0]:
            for _ in range(400):
                s = rng.integers(N_LIB); add('length', cond, L, 1, s, lib[s])
            for _ in range(400):
                add('length', cond, L, 0, None, rng.choice(neg_test))
    # library-size scaling: positives from the first 100 songs (present in every library size)
    for cond in ['clean', 'mic']:
        for _ in range(300):
            s = rng.integers(100); add('scale', cond, 5.0, 1, s, lib[s])
        for _ in range(500):
            add('scale', cond, 5.0, 0, None, rng.choice(neg_test))
    # sequential decisions: 10 s queries, evaluated after every second
    for cond in ['clean', 'pink5', 'mic']:
        for _ in range(300):
            s = rng.integers(N_LIB); add('seq', cond, 10.0, 1, s, lib[s])
        for _ in range(500):
            add('seq', cond, 10.0, 0, None, rng.choice(neg_test))
    return specs


def query_audio(spec):
    """Deterministic degraded query at 44.1 kHz, int16 scale, float64."""
    x = data.load44(spec['track'])[: int(CLIP * 44100)]
    mono = x.mean(axis=1).astype(np.float64)
    rng = np.random.default_rng(spec['seed'])
    c, L = spec['cond'], spec['L']
    fac = 1 + float(c[5:-1]) / 100 if c.startswith('speed') else 1.0
    s = int(spec['start'] * 44100)
    seg = mono[s: s + int(round(L * fac * 44100))]
    y = augment.apply(seg, 44100, c, rng)[: int(L * 44100)] * 32767.0
    peak = np.abs(y).max() if len(y) else 0.0
    if peak > 32767:                      # a real 16-bit recording cannot exceed full scale: automatic gain
        y = y * (32767 / peak)
    return y


def _v1_query(y):
    y = y[: (len(y) // 2048) * 2048]
    S = v1.spectrogram_original(v1.preprocess_original(v1.to_mono_original(y)))   # mono input is halved, as in the app
    out = {}
    for fixed, tag in ((False, 'v1o'), (True, 'v1f')):
        t, f = v1.peaks_original(S, fixed=fixed)
        k, ta = v1.hashes_original(t, f)
        out[tag] = (k, ta, np.repeat(np.arange(len(k) // 5), 5))
    return out


def prepare_one(spec):
    try:
        y = query_audio(spec)
        out = {'id': spec['id']}
        if spec['grid'] == 'seq':
            out['v2_prefix'] = []
            for T in range(1, 11):
                k, t, a, nf = v2.fingerprint(y[: int(T * 44100)], 44100)
                out['v2_prefix'].append((k, t, a, nf))
            return out
        out.update(_v1_query(y))
        k, t, a, nf = v2.fingerprint(y, 44100)
        out['v2'] = (k, t, a, nf)
        return out
    except Exception as e:
        return {'id': spec['id'], 'error': repr(e)}


def prepare(workers=10):
    os.makedirs(WORK, exist_ok=True)
    specs = make_specs()
    pickle.dump(specs, open(os.path.join(WORK, 'specs.pkl'), 'wb'))
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(prepare_one, specs, chunksize=8)):
            out[r['id']] = r
            if i % 2000 == 0:
                print(i, len(specs), f'{time.time() - t0:.0f}s', flush=True)
    errs = [r for r in out.values() if 'error' in r]
    print('errors', len(errs), errs[:3])
    pickle.dump(out, open(os.path.join(WORK, 'prepared.pkl'), 'wb'), protocol=4)


# ------------------------------------------------------------------ classical matching
def _decide(idx, q, unit, song_frames, kq, original_rules_seconds=None, pairs_too=False, pairs=None):
    k, t, a = q[:3]
    r = {}
    if pairs is None:
        t0 = time.perf_counter()
        pairs = idx.lookup(k, t)
        r['lookup_ms'] = (time.perf_counter() - t0) * 1000
    if original_rules_seconds is not None:
        t0 = time.perf_counter()
        sid, score, _ = match_original_rules(idx, k, t, original_rules_seconds, pairs=pairs)
        r['rule_song'], r['rule_score'], r['rule_ms'] = sid, score, (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    votes = offset_votes(idx, k, t, qanchor=a, unit=unit, radius=1, pairs=pairs)
    ks = idx.song_frames[votes['song']] if len(votes['song']) else np.zeros(0)
    p, top, c, second = stats.family_pvalue(votes, ks, kq)
    r['ms'] = (time.perf_counter() - t0) * 1000 + r.get('lookup_ms', 0)
    r.update({'p': p, 'top': top, 'score': c, 'second': second, 'M': int(votes['M'].sum()), 'n_hash': len(k),
              'n_pairs': int(len(pairs[0]))})
    # p-values under quasi-Poisson over-dispersion factors (for the calibration study)
    r['p_phi'] = [stats.family_pvalue(votes, ks, kq, phi=phi)[0] for phi in PHIS]
    if pairs_too:
        vp = offset_votes(idx, k, t, qanchor=None, unit=unit, radius=1, pairs=pairs)
        r['p_pairs'] = stats.family_pvalue(vp, idx.song_frames[vp['song']] if len(vp['song']) else np.zeros(0), kq)[0]
    return r


_IDX = None
SIZES = [100, 300, 1000, 2000, 4000]


def _classical_one(spec):
    global _IDX
    if _IDX is None:
        _IDX = {tag: HashIndex.load_mmap(os.path.join(WORK, f'index_{tag}')) for tag in ('v1o', 'v1f', 'v2')}
    q = _PREP_CACHE.get(spec['id']) if _PREP_CACHE else None
    if q is None or 'error' in q:
        return spec['id'], None
    idx = _IDX
    L = spec['L']
    if spec['grid'] == 'seq':
        ps = []
        for qq in q['v2_prefix']:
            r = _decide(idx['v2'], qq, 1, V2_SONG_FRAMES, qq[3])
            ps.append((r['p'], r['top'], r['score'], r['p_phi']))
        return spec['id'], {'seq': ps}
    kq1 = L / v1.FRAME_SEC
    libs = SIZES if spec['grid'] == 'scale' else [4000]
    full = {tag: idx[tag].lookup(*(q[tag][:2])) for tag in ('v1o', 'v1f', 'v2')}
    out = {}
    for n in libs:
        o = {}
        for tag in ('v1o', 'v1f', 'v2'):
            s_, d_, qi_ = full[tag]
            m = s_ < n
            pr = (s_[m], d_[m], qi_[m]) if n < 4000 else full[tag]
            if tag == 'v2':
                o[tag] = _decide(idx[tag], q[tag], 1, V2_SONG_FRAMES, q[tag][3], pairs_too=(n == 4000), pairs=pr)
            else:
                o[tag] = _decide(idx[tag], q[tag], V1_UNIT, V1_SONG_FRAMES, kq1, pairs=pr,
                                 original_rules_seconds=min(10.0, L) if tag == 'v1o' else None)
        out[n] = o
    return spec['id'], out


_PREP_CACHE = None


def _init_worker(chunk_path):
    global _PREP_CACHE
    _PREP_CACHE = pickle.load(open(chunk_path, 'rb'))


def classical(workers=12):
    specs = pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb'))
    prep = pickle.load(open(os.path.join(WORK, 'prepared.pkl'), 'rb'))
    for tag in ('v1o', 'v1f', 'v2'):
        folder = os.path.join(WORK, f'index_{tag}')
        if not os.path.exists(os.path.join(folder, 'keys.npy')):
            HashIndex.load(os.path.join(WORK, f'index_{tag}.npz')).save_npy(folder)
    # split the prepared queries into one file per worker batch so each process loads only its share
    t0 = time.time()
    results = {}
    nchunks = workers * 4
    chunks = [specs[i::nchunks] for i in range(nchunks)]
    paths = []
    for c, ch in enumerate(chunks):
        pth = os.path.join(WORK, f'prep_chunk_{c}.pkl')
        pickle.dump({s['id']: prep[s['id']] for s in ch}, open(pth, 'wb'), protocol=4)
        paths.append(pth)
    del prep
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_run_chunk, pth, ch) for pth, ch in zip(paths, chunks)]
        for i, f in enumerate(futs):
            results.update(f.result())
            print('chunk', i + 1, '/', len(futs), f'{time.time() - t0:.0f}s', flush=True)
    for pth in paths:
        os.remove(pth)
    pickle.dump(results, open(os.path.join(WORK, 'classical.pkl'), 'wb'), protocol=4)
    print('done', f'{time.time() - t0:.0f}s')


def _run_chunk(path, chunk):
    _init_worker(path)
    return dict(_classical_one(s) for s in chunk)


# ------------------------------------------------------------------ neural
_MODEL = None


def _neural_query(spec):
    global _MODEL
    import torch
    from research import neural
    if _MODEL is None:
        torch.set_num_threads(1)
        _MODEL = neural.FingerprintNet()
        _MODEL.load_state_dict(torch.load(os.path.join(WORK, 'neural.pt')))
        _MODEL.eval()
    y = query_audio(spec) / 32767.0
    y11 = data.resample(y.astype(np.float32), 44100, 11025)
    if spec['grid'] == 'seq':
        return spec['id'], None
    return spec['id'], neural.embed_segments(_MODEL, y11).numpy().astype(np.float16)


def neural_eval(workers=8):
    import torch
    from research import neural
    torch.set_num_threads(16)
    specs = pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb'))
    model = neural.FingerprintNet()
    model.load_state_dict(torch.load(os.path.join(WORK, 'neural.pt')))
    _, sp = names()
    _, mm = data.cached()
    t0 = time.time()
    embs, tr, sg = [], [], []
    for song, i in enumerate(sp['library'][:N_LIB]):
        e = neural.embed_segments(model, mm[i].astype(np.float32) / 32768)
        embs.append(e); tr.append(torch.full((len(e),), song)); sg.append(torch.arange(len(e)))
    nidx = neural.NeuralIndex(torch.cat(embs), torch.cat(tr), torch.cat(sg))
    lib_sec = time.time() - t0
    print('library embedded', nidx.emb.shape, f'{lib_sec:.0f}s', flush=True)
    torch.save({'emb': nidx.emb, 'track': nidx.track, 'seg': nidx.seg}, os.path.join(WORK, 'neural_index.pt'))
    qemb = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (qid, e) in enumerate(ex.map(_neural_query, specs, chunksize=16)):
            qemb[qid] = e
            if i % 4000 == 0:
                print('embedded', i, f'{time.time() - t0:.0f}s', flush=True)
    sizes = [100, 300, 1000, 2000, 4000]
    subs = {n: nidx.subset(n) for n in sizes}
    results = {}
    t1 = time.time()
    for spec in specs:
        e = qemb.get(spec['id'])
        if e is None:
            continue
        q = torch.from_numpy(e.astype(np.float32))
        libs = sizes if spec['grid'] == 'scale' else [4000]
        results[spec['id']] = {n: subs[n].search(q) for n in libs}
    search_ms = (time.time() - t1) / max(1, len(results)) * 1000
    pickle.dump({'results': results, 'lib_embed_sec': lib_sec, 'search_ms': search_ms},
                open(os.path.join(WORK, 'neural_results.pkl'), 'wb'))
    print('neural done', f'search {search_ms:.1f} ms/query')


def _patch_one(spec):
    y = query_audio(spec)
    return spec['id'], _v1_query(y)


def patch_v1(workers=14):
    """Recompute the v1 fingerprints of noisy queries after adding the 16-bit gain limit (fixes float16 overflow)."""
    specs = pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb'))
    prep = pickle.load(open(os.path.join(WORK, 'prepared.pkl'), 'rb'))
    todo = [s for s in specs if s['grid'] != 'seq' and s['cond'] in ('pink20', 'pink10', 'pink5', 'pink0', 'reverb', 'mic')]
    with ProcessPoolExecutor(workers) as ex:
        for qid, out in ex.map(_patch_one, todo, chunksize=8):
            prep[qid].update(out)
    pickle.dump(prep, open(os.path.join(WORK, 'prepared.pkl'), 'wb'), protocol=4)
    print('patched', len(todo))


# ------------------------------------------------------------------ extra conformal set
def conformal_specs(start_id):
    """2500 extra non-library queries (500 per length) for conformal thresholds (from the calibration
    pool of negative songs, disjoint from the test negatives)."""
    tracks, sp = names()
    neg_cal = sp['negative'][:500]
    rng = np.random.default_rng(SEED + 1)
    out = []
    for L in [1.0, 2.0, 3.0, 5.0, 10.0]:
        for _ in range(500):
            cond = str(rng.choice(augment.CONDITIONS))
            fac = 1 + float(cond[5:-1]) / 100 if cond.startswith('speed') else 1.0
            out.append({'id': start_id + len(out), 'grid': 'conf', 'cond': cond, 'L': L, 'label': 0, 'song': -1,
                        'track': tracks[int(rng.choice(neg_cal))], 'start': float(rng.uniform(0, CLIP - L * fac - 0.2)),
                        'seed': int(rng.integers(2**31))})
    return out


def add_conformal_set(workers=12):
    import torch
    from research import neural
    specs = pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb'))
    if any(s['grid'] == 'conf' for s in specs):
        print('already added'); return
    new = conformal_specs(max(s['id'] for s in specs) + 1)
    with ProcessPoolExecutor(workers) as ex:
        prep = {r['id']: r for r in ex.map(prepare_one, new, chunksize=8)}
    pth = os.path.join(WORK, 'prep_chunk_conf.pkl')
    pickle.dump(prep, open(pth, 'wb'), protocol=4)
    chunks = [new[i::workers] for i in range(workers)]
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_run_chunk, [pth] * len(chunks), chunks):
            res.update(r)
    os.remove(pth)
    cl = pickle.load(open(os.path.join(WORK, 'classical.pkl'), 'rb'))
    cl.update(res)
    pickle.dump(cl, open(os.path.join(WORK, 'classical.pkl'), 'wb'), protocol=4)
    # neural
    z = torch.load(os.path.join(WORK, 'neural_index.pt'))
    nidx = neural.NeuralIndex(z['emb'], z['track'], z['seg'])
    nr = pickle.load(open(os.path.join(WORK, 'neural_results.pkl'), 'rb'))
    with ProcessPoolExecutor(workers) as ex:
        for qid, e in ex.map(_neural_query, new, chunksize=16):
            nr['results'][qid] = {4000: nidx.search(torch.from_numpy(e.astype(np.float32)))}
    pickle.dump(nr, open(os.path.join(WORK, 'neural_results.pkl'), 'wb'))
    pickle.dump(specs + new, open(os.path.join(WORK, 'specs.pkl'), 'wb'))
    print('conformal set added', len(new))


def rerun_grid(grid='seq', workers=12):
    specs = [s for s in pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb')) if s['grid'] == grid]
    prep = pickle.load(open(os.path.join(WORK, 'prepared.pkl'), 'rb'))
    pth = os.path.join(WORK, f'prep_chunk_{grid}.pkl')
    pickle.dump({s['id']: prep[s['id']] for s in specs}, open(pth, 'wb'), protocol=4)
    del prep
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_run_chunk, [pth] * workers, [specs[i::workers] for i in range(workers)]):
            res.update(r)
    os.remove(pth)
    cl = pickle.load(open(os.path.join(WORK, 'classical.pkl'), 'rb'))
    cl.update(res)
    pickle.dump(cl, open(os.path.join(WORK, 'classical.pkl'), 'wb'), protocol=4)
    print('rerun', grid, len(res))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'build': build, 'prepare': prepare, 'patch_v1': patch_v1, 'classical': classical, 'neural': neural_eval,
     'conformal': add_conformal_set, 'rerun_seq': rerun_grid}[cmd]()
