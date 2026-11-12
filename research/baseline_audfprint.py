"""External baseline: audfprint (Dan Ellis, github.com/dpwe/audfprint), a widely used open-source
landmark fingerprinter, run on exactly the same library and queries as our engines.

Default settings (density 20, 20-bit hashes, bucket depth 100, match threshold 5 aligned hashes),
queries analysed with --shifts 4.
Audio is passed as arrays (mono 11025 Hz), so audfprint's ffmpeg reader is not used.
Source is downloaded to data/ext/audfprint-master (not part of this project).

python -m research.baseline_audfprint
"""
import os, sys, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from research import data
from research.bench import WORK, query_audio, N_LIB

sys.path.insert(0, os.path.join(data.DATA, 'ext', 'audfprint-master'))
import audfprint_analyze, audfprint_match, hash_table   # noqa: E402

_AN = None
QUERY_SHIFTS = 4          # audfprint's recommended query option against frame-alignment effects


def _analyzer():
    global _AN
    if _AN is None:
        _AN = audfprint_analyze.Analyzer()
    return _AN


def hashes_of(x11, shifts=1):
    """Same as audfprint's wavfile2hashes: with shifts > 1 the query is analysed at several
    sub-frame offsets (audfprint's --shifts option for queries) and duplicate hashes are removed."""
    an = _analyzer()
    d = np.asarray(x11, dtype=np.float64)
    hs = []
    for k in range(max(1, shifts)):
        peaks = an.find_peaks(d[int(k / max(1, shifts) * an.n_hop):], 11025)
        hs.append(audfprint_analyze.landmarks2hashes(an.peaks2landmarks(peaks)))
    h = np.concatenate(hs) if hs else np.zeros((0, 2), np.int32)
    return np.unique(h, axis=0) if len(h) else h


def _lib_song(i):
    _, mm = data.cached()
    return hashes_of(mm[i].astype(np.float64) / 32768)


def _query(spec):
    y = query_audio(spec) / 32767.0
    return spec['id'], hashes_of(data.resample(y.astype(np.float32), 44100, 11025), shifts=QUERY_SHIFTS)


def main(workers=12):
    _, sp = data.cached()[0], data.splits()
    t0 = time.time()
    ht = hash_table.HashTable(hashbits=20, depth=100, maxtime=16384)
    with ProcessPoolExecutor(workers) as ex:
        for song, h in enumerate(ex.map(_lib_song, list(sp['library'][:N_LIB]), chunksize=16)):
            ht.store(str(song), h)
    build_sec = time.time() - t0
    print('audfprint library built', ht.totalhashes(), f'{build_sec:.0f}s', flush=True)
    specs = [s for s in pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb')) if s['grid'] in ('conf', 'main', 'length')]
    matcher = audfprint_match.Matcher()
    matcher.threshcount = 1          # return every candidate; thresholds are applied in the analysis
    out = {}
    t1 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (qid, h) in enumerate(ex.map(_query, specs, chunksize=16)):
            res = matcher.match_hashes(ht, h) if len(h) else np.zeros((0, 7))
            if len(res):
                top = int(ht.names[int(res[0, 0])])
                second = float(res[1, 1]) if len(res) > 1 else 0.0
                out[qid] = (top, float(res[0, 1]), second)
            else:
                out[qid] = (-1, 0.0, 0.0)
            if i % 4000 == 0:
                print('queries', i, len(specs), f'{time.time() - t1:.0f}s', flush=True)
    pickle.dump({'results': out, 'build_sec': build_sec, 'query_sec': (time.time() - t1) / len(specs)},
                open(os.path.join(WORK, 'audfprint_results.pkl'), 'wb'))
    print('done')


if __name__ == '__main__':
    main()
