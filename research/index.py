"""Inverted index of fingerprint hashes + vectorised matching.

The original stores a Python dict {address string: [couple strings]}. Here the same data
lives in three sorted numpy arrays (key, song, anchor time), so a lookup is a binary
search (np.searchsorted) and voting is done with np.unique. This scales to thousands of
songs and gives identical votes.
"""
import os
import numpy as np



class HashIndex:
    def __init__(self, keys, songs, times, song_frames):
        self.keys, self.songs, self.times = keys, songs, times
        self.song_frames = np.asarray(song_frames, dtype=np.int64)   # length of each song in time units

    @classmethod
    def build(cls, per_song, song_frames):
        """per_song: list of (keys, times) for song ids 0..n-1 (in order)."""
        keys = np.concatenate([k for k, _ in per_song]) if per_song else np.zeros(0, np.int64)
        times = np.concatenate([t for _, t in per_song]) if per_song else np.zeros(0, np.int64)
        songs = np.concatenate([np.full(len(k), s, np.int32) for s, (k, _) in enumerate(per_song)]) \
            if per_song else np.zeros(0, np.int32)
        order = np.argsort(keys, kind='stable')          # stable: keeps insertion order per key
        return cls(keys[order].astype(np.int64), songs[order], times[order].astype(np.int32), song_frames)

    @property
    def n_songs(self):
        return len(self.song_frames)

    @property
    def nbytes(self):
        return self.keys.nbytes + self.songs.nbytes + self.times.nbytes

    def save(self, path):
        np.savez(path, keys=self.keys, songs=self.songs, times=self.times, song_frames=self.song_frames)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['keys'], z['songs'], z['times'], z['song_frames'])

    def save_npy(self, folder):
        os.makedirs(folder, exist_ok=True)
        for name in ('keys', 'songs', 'times', 'song_frames'):
            np.save(os.path.join(folder, name + '.npy'), getattr(self, name))

    @classmethod
    def load_mmap(cls, folder):
        """Memory-mapped load: many processes share one copy of the index in RAM."""
        g = lambda n: np.load(os.path.join(folder, n + '.npy'), mmap_mode='r')
        return cls(g('keys'), g('songs'), g('times'), np.asarray(g('song_frames')))

    def subset(self, n):
        """Index restricted to songs 0..n-1 (used for the library-size experiments)."""
        m = self.songs < n
        return HashIndex(self.keys[m], self.songs[m], self.times[m], self.song_frames[:n])

    def lookup(self, qkeys, qtimes):
        """All (query hash, database hash) pairs with equal keys.

        Returns song ids, delta = t_db - t_query, and the index of the query hash for each pair.
        """
        lo = np.searchsorted(self.keys, qkeys, 'left')
        hi = np.searchsorted(self.keys, qkeys, 'right')
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            z = np.zeros(0, np.int64)
            return z, z, z
        qi = np.repeat(np.arange(len(qkeys)), cnt)
        starts = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
        pos = starts + np.arange(total)
        return self.songs[pos].astype(np.int64), self.times[pos].astype(np.int64) - qtimes[qi], qi


_D_OFF = np.int64(1) << 20          # offsets / times are packed next to the song id in one int64
_S_SHIFT = 21


def _pack(song, val):
    return (song << _S_SHIFT) | (val + _D_OFF)


def match_original_rules(index, qkeys, qtimes, seconds, pairs=None):
    """DBModule.SearchDatabase, rule for rule.

    (a) a database couple (song, anchor time) must be returned >= 4 times,
    (b) a song needs >= seconds*50*0.1 such couples,
    (c) among those songs, the most common time offset must occur > 39 times; best song wins.
    Returns (song id or -1, best offset count, details).
    """
    songs, delta, qi = index.lookup(qkeys, qtimes) if pairs is None else pairs
    if len(songs) == 0:
        return -1, 0, {}
    tdb = delta + qtimes[qi]
    uc, cc = np.unique(_pack(songs, tdb), return_counts=True)
    good = uc[cc >= 4]
    gsong, gcount = np.unique(good >> _S_SHIFT, return_counts=True)
    cand = gsong[~(gcount < seconds * 50 * 0.1)]          # same float expression as the original
    if len(cand) == 0:
        return -1, 0, {'candidates': 0}
    # most common delta per candidate song, over ALL returned pairs (as in CheckTimeCoherency)
    m = np.isin(songs, cand)
    usd, csd = np.unique(_pack(songs[m], delta[m]), return_counts=True)
    s_of = usd >> _S_SHIFT
    starts = np.concatenate([[0], np.flatnonzero(np.diff(s_of)) + 1])
    song_ids, best = s_of[starts], np.maximum.reduceat(csd, starts)
    # the original iterates songs in dict insertion order = first appearance among the returned pairs
    us, first = np.unique(songs, return_index=True)
    order = np.argsort(first[np.searchsorted(us, song_ids)], kind='stable')
    highest, key = 1, -1
    for n in order:
        if highest < best[n]:
            highest = best[n]
            if highest > 39:
                key = song_ids[n]
    return int(key), int(best.max()), {'candidates': len(cand)}


def offset_votes(index, qkeys, qtimes, qanchor=None, unit=1.0, radius=1, pairs=None):
    """Offset voting used by the improved engines and the statistics.

    delta is converted to the time grid (`unit` = database time units per grid step) and the
    votes in delta-radius .. delta+radius are summed (fixes votes split by rounding / frame
    misalignment). If qanchor is given, each query anchor votes at most once per (song, delta),
    which makes votes close to independent (used by the Poisson model).

    Returns dict with per-song arrays over songs that got any vote:
      song, score (best merged count), best_delta, M (total votes).
    """
    songs, delta, qi = index.lookup(qkeys, qtimes) if pairs is None else pairs
    if len(songs) == 0:
        e = np.zeros(0, np.int64)
        return {'song': e, 'score': e, 'best_delta': e, 'M': e}
    d = np.rint(delta / unit).astype(np.int64)
    comb = _pack(songs, d)
    if qanchor is not None:
        qa = np.asarray(qanchor, dtype=np.int64)[qi]
        if qa.max() < (1 << 14):
            comb = np.unique((comb << 14) | qa) >> 14            # one vote per (anchor, song, delta)
        else:
            comb = np.unique(np.stack([comb, qa], 1), axis=0)[:, 0]
    uc, cc = np.unique(comb, return_counts=True)
    merged = cc.copy()
    for k in range(1, radius + 1):
        for sign in (-1, 1):
            tgt = uc + sign * k
            pos = np.minimum(np.searchsorted(uc, tgt), len(uc) - 1)
            merged = merged + np.where(uc[pos] == tgt, cc[pos], 0)
    s_of = uc >> _S_SHIFT
    starts = np.concatenate([[0], np.flatnonzero(np.diff(s_of)) + 1])
    score = np.maximum.reduceat(merged, starts)
    M = np.add.reduceat(cc, starts)
    order = np.lexsort((-merged, s_of))                         # best window first within each song
    best_delta = (uc[order[starts]] & ((np.int64(1) << _S_SHIFT) - 1)) - _D_OFF
    return {'song': s_of[starts], 'score': score, 'best_delta': best_delta, 'M': M}
