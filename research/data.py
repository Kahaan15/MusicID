"""Dataset access: FMA small (8000 x 30 s MP3, Creative Commons) read straight from the zip.

Also caches every track as mono int16 at 11025 Hz in one memory-mapped file
(data/fma11k.int16) for fast random access by the v2 engine and the neural model.
"""
import io, os, zipfile, json
import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data')
ZIP = os.path.join(DATA, 'fma_small.zip')
CACHE = os.path.join(DATA, 'fma11k.int16')
META = os.path.join(DATA, 'fma11k.json')
CLIP_SEC = 29.0                   # every cached track is trimmed to 29 s (FMA clips are ~30 s)
SR11 = 11025
N11 = int(CLIP_SEC * SR11)

_zip = None


def _z():
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(ZIP)
    return _zip


def list_tracks():
    return sorted(n for n in _z().namelist() if n.endswith('.mp3'))


def decode(name):
    """MP3 from the zip -> float32 array (n, channels), sample rate."""
    data = _z().read(name)
    x, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    return x, sr


def resample(x, sr, target):
    if sr == target:
        return x
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, int(sr) // g, axis=0).astype(np.float32)


def load44(name):
    """Track as the app would see it: 44.1 kHz, (n, channels) float32 in [-1, 1]."""
    x, sr = decode(name)
    return resample(x, sr, 44100)


def mono11(x44):
    return resample(x44.mean(axis=1) if x44.ndim == 2 else x44, 44100, SR11)


def _cache_one(name):
    try:
        x44 = load44(name)
        if len(x44) < CLIP_SEC * 44100:
            return name, None
        y = mono11(x44[:int(CLIP_SEC * 44100)])[:N11]
        if len(y) < N11 or np.abs(y).max() < 1e-4:
            return name, None
        return name, np.clip(np.rint(y * 32767), -32768, 32767).astype(np.int16)
    except Exception:
        return name, None


def build_cache(workers=16):
    """Decode all tracks once (parallel). Broken / short / silent tracks are dropped."""
    from concurrent.futures import ProcessPoolExecutor
    names = list_tracks()
    mm = np.lib.format.open_memmap(CACHE + '.npy', mode='w+', dtype=np.int16, shape=(len(names), N11))
    ok = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (name, y) in enumerate(ex.map(_cache_one, names, chunksize=8)):
            if y is not None:
                mm[len(ok)] = y
                ok.append(name)
    mm.flush()
    del mm
    json.dump({'tracks': ok, 'n_total': len(names)}, open(META, 'w'))
    return ok


def cached():
    meta = json.load(open(META))
    mm = np.load(CACHE + '.npy', mmap_mode='r')
    return meta['tracks'], mm


def splits(seed=0):
    """Fixed split of the valid tracks: library pool (4000), negative pool (1000), training pool (rest)."""
    tracks, _ = cached()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tracks))
    return {'library': order[:4000], 'negative': order[4000:5000], 'train': order[5000:]}


# ---------------------------------------------------------------- metadata (titles / artists)
META_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'
TRACKS_CSV = os.path.join(DATA, 'fma_tracks.csv')


class _HTTPFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP range requests (enough for zipfile)."""
    def __init__(self, url):
        import requests
        self.url, self.pos, self.s = url, 0, requests.Session()
        self.size = int(self.s.head(url, allow_redirects=True).headers['Content-Length'])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b''
        r = self.s.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'}, timeout=120)
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def fetch_tracks_csv(member='fma_metadata/tracks.csv', workers=16):
    """Download only tracks.csv out of the remote metadata zip (parallel byte ranges, then inflate)."""
    import zlib, struct, requests
    from concurrent.futures import ThreadPoolExecutor
    if os.path.exists(TRACKS_CSV):
        return TRACKS_CSV
    hf = _HTTPFile(META_URL)
    info = zipfile.ZipFile(hf).getinfo(member)
    start = info.header_offset
    end = start + 30 + len(info.filename) + 1024 + info.compress_size
    chunk = 2 * 1024 * 1024
    ranges = [(a, min(a + chunk, end) - 1) for a in range(start, end, chunk)]

    def get(r):
        for _ in range(6):
            try:
                x = requests.get(META_URL, headers={'Range': f'bytes={r[0]}-{r[1]}'}, timeout=120).content
                if len(x) == r[1] - r[0] + 1:
                    return x
            except Exception:
                pass
        raise RuntimeError('range failed')

    with ThreadPoolExecutor(workers) as ex:
        blob = b''.join(ex.map(get, ranges))
    n, m = struct.unpack('<HH', blob[26:30])
    raw = blob[30 + n + m: 30 + n + m + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data_ = zlib.decompress(raw, -15)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        import bz2
        data_ = bz2.decompress(raw)
    else:
        data_ = raw
    open(TRACKS_CSV, 'wb').write(data_)
    return TRACKS_CSV


def track_metadata():
    """{track id: {title, artist, album, year, genre, license}} for the FMA tracks."""
    import pandas as pd
    df = pd.read_csv(fetch_tracks_csv(), index_col=0, header=[0, 1], low_memory=False)
    out = {}
    for tid, row in df.iterrows():
        date = str(row.get(('album', 'date_released'), '') or '')
        if date in ('', 'nan'):
            date = str(row.get(('track', 'date_created'), '') or '')
        out[int(tid)] = {'title': str(row[('track', 'title')]), 'artist': str(row[('artist', 'name')]),
                         'album': str(row[('album', 'title')]), 'year': date[:4] if date[:4].isdigit() else '',
                         'genre': str(row[('track', 'genre_top')]), 'license': str(row[('track', 'license')])}
    return out
