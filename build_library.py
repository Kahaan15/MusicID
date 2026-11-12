"""Build the app's song library (library/index.npz + library/songs.json).

Contents: song.wav (Never Gonna Give You Up, Rick Astley) + every valid track of the
FMA-small dataset (Creative Commons music, titles/artists from FMA's tracks.csv).

python build_library.py          (needs data/ from research/download_fma.py + research.data.build_cache)
"""
import os, sys, json, time
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

import EngineModule
from research import data, fingerprint_v2 as v2


def _fp(i):
    _, mm = data.cached()
    k, ta, aid, nf = v2.fingerprint_11k(mm[i].astype(np.float64) / 32768)
    return k.astype(np.uint32), ta.astype(np.uint16), nf


def main():
    t0 = time.time()
    keys, songs, times, frames, meta = [], [], [], [], []

    # 1) the original song of the project
    x, sr = sf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'song.wav'), dtype='int16')
    k, ta, aid, nf = v2.fingerprint(x, sr)
    keys.append(k.astype(np.uint32)); times.append(ta.astype(np.uint16)); songs.append(np.zeros(len(k), np.uint16)); frames.append(nf)
    meta.append({'id': 0, 'title': 'Never Gonna Give You Up', 'artist': 'Rick Astley', 'album': 'Single', 'year': '1987'})

    # 2) the FMA tracks
    tracks, _ = data.cached()
    md = data.track_metadata()
    with ProcessPoolExecutor(8) as ex:
        for i, (k, ta, nf) in enumerate(ex.map(_fp, range(len(tracks)), chunksize=32)):
            sid = len(meta)
            tid = int(os.path.basename(tracks[i])[:-4])
            m = md.get(tid, {})
            keys.append(k); times.append(ta); songs.append(np.full(len(k), sid, np.uint16)); frames.append(nf)
            meta.append({'id': sid, 'title': m.get('title', f'FMA track {tid}'), 'artist': m.get('artist', 'Unknown'),
                         'album': m.get('album', ''), 'year': m.get('year', ''), 'genre': m.get('genre', ''),
                         'license': m.get('license', ''), 'fma_id': tid})
    keys, songs, times = np.concatenate(keys), np.concatenate(songs), np.concatenate(times)
    order = np.argsort(keys, kind='stable')
    os.makedirs(EngineModule.LIB_DIR, exist_ok=True)
    phi = EngineModule.PHI
    cal = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research', 'results', 'summary.json')
    if os.path.exists(cal):
        phi = float(json.load(open(cal))['calibration']['phi_hat'])
    np.savez(EngineModule.INDEX_PATH, keys=keys[order], songs=songs[order], times=times[order],
             song_frames=np.array(frames, np.int64), phi=phi)
    json.dump(meta, open(EngineModule.SONGS_PATH, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{len(meta)} songs, {len(keys):,} hashes, {(keys.nbytes + songs.nbytes + times.nbytes) / 2**20:.0f} MB, '
          f'phi={phi}, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
