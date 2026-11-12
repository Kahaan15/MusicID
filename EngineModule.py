"""MusicID engine used by the app (GUIModule) and add_song.py.

* fingerprint: improved classical v2 (research/fingerprint_v2.py)
* index: compact sorted arrays (library/index.npz), song details in library/songs.json
* decision: family-wise Poisson p-value (research/stats.py). While recording, the app looks
  again every second; the error budget alpha is split over all looks (Bonferroni), so the
  overall chance of naming a song that is not in the library stays below alpha.

The original engine (AudioModule + DBModule) is kept unchanged for reference and tests.
"""
import os, json
import numpy as np

from research import fingerprint_v2 as v2, stats
from research.index import HashIndex, offset_votes

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'library')
INDEX_PATH = os.path.join(LIB_DIR, 'index.npz')
SONGS_PATH = os.path.join(LIB_DIR, 'songs.json')
ALPHA = 0.01                # overall false-match budget
PHI = 1.0                   # over-dispersion factor (set from the calibration study in research/results)


class Engine:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.songs = json.load(open(SONGS_PATH, encoding='utf-8')) if os.path.exists(SONGS_PATH) else []
        if os.path.exists(INDEX_PATH):
            z = np.load(INDEX_PATH)
            self.index = HashIndex(z['keys'], z['songs'], z['times'], z['song_frames'])
            self.phi = float(z['phi']) if 'phi' in z else PHI
        else:
            self.index = HashIndex(np.zeros(0, np.uint32), np.zeros(0, np.uint16), np.zeros(0, np.uint16), [])
            self.phi = PHI

    @property
    def n_songs(self):
        return len(self.songs)

    # ------------------------------------------------------------ identify
    def identify(self, audio, sr, looks=1):
        """Identify a recording. `looks` = total number of checks planned for this recording
        (the app checks every second while recording); each check uses alpha / looks, so the
        overall false-match probability over all checks stays <= alpha (Bonferroni).

        Returns dict: matched, song (dict) or None, p_value, score, offset_sec, confidence.
        """
        k, t, a, nf = v2.fingerprint(audio, sr)
        if len(k) == 0 or self.index.n_songs == 0:
            return {'matched': False, 'song': None, 'p_value': 1.0, 'score': 0, 'offset_sec': None}
        votes = offset_votes(self.index, k.astype(self.index.keys.dtype), t, qanchor=a, unit=1, radius=1)
        if len(votes['song']) == 0:
            return {'matched': False, 'song': None, 'p_value': 1.0, 'score': 0, 'offset_sec': None}
        p, top, score, second = stats.family_pvalue(votes, self.index.song_frames[votes['song']], nf, phi=self.phi)
        best = int(np.flatnonzero(votes['song'] == top)[0])
        offset = float(votes['best_delta'][best] * v2.FRAME_SEC)
        matched = p < self.alpha / max(1, looks)
        return {'matched': bool(matched), 'song': self.songs[top] if matched else None, 'p_value': p,
                'score': score, 'second_score': second, 'offset_sec': max(0.0, offset), 'top_id': top}

    # ------------------------------------------------------------ library management
    def add_song(self, audio, sr, meta):
        """Fingerprint a song and add it to the library (meta: title, artist, album, year)."""
        k, t, a, nf = v2.fingerprint(audio, sr)
        sid = len(self.songs)
        idx = self.index
        keys = np.concatenate([idx.keys.astype(np.uint32), k.astype(np.uint32)])
        songs = np.concatenate([idx.songs.astype(np.uint16), np.full(len(k), sid, np.uint16)])
        times = np.concatenate([idx.times.astype(np.uint16), t.astype(np.uint16)])
        order = np.argsort(keys, kind='stable')
        self.index = HashIndex(keys[order], songs[order], times[order], np.concatenate([idx.song_frames, [nf]]))
        self.songs.append(dict(meta, id=sid))
        return sid

    def find(self, title):
        return [s for s in self.songs if s.get('title', '').lower() == title.lower()]

    def save(self):
        os.makedirs(LIB_DIR, exist_ok=True)
        np.savez(INDEX_PATH, keys=self.index.keys, songs=self.index.songs, times=self.index.times,
                 song_frames=self.index.song_frames, phi=self.phi)
        json.dump(self.songs, open(SONGS_PATH, 'w', encoding='utf-8'), ensure_ascii=False)


def format_time(sec):
    sec = int(round(sec))
    return f'{sec // 60}:{sec % 60:02d}'
