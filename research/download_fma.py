"""Download fma_small.zip (8000 x 30 s MP3 clips, CC-licensed) from the official FMA host.

The host is slow per connection, so the file is fetched as many byte ranges in parallel
and written into a pre-allocated file. Finished ranges are remembered, so it can resume.
"""
import os, sys, json, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
SIZE = 7679594875
CHUNK = 16 * 1024 * 1024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fma_small.zip')
STATE = OUT + '.done.json'

def main(workers=16):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(OUT):
        with open(OUT, 'wb') as f:
            f.truncate(SIZE)
    done = set(json.load(open(STATE))) if os.path.exists(STATE) else set()
    starts = [s for s in range(0, SIZE, CHUNK) if s not in done]
    lock = threading.Lock()
    t0 = time.time(); got = [0]

    def fetch(start):
        end = min(start + CHUNK, SIZE) - 1
        for attempt in range(8):
            try:
                r = requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    with lock:
                        with open(OUT, 'r+b') as f:
                            f.seek(start); f.write(r.content)
                        done.add(start); got[0] += len(r.content)
                        json.dump(sorted(done), open(STATE, 'w'))
                    return start
            except Exception as e:
                pass
            time.sleep(2 + attempt * 3)
        raise RuntimeError(f'chunk {start} failed')

    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(fetch, s) for s in starts]
        for i, fu in enumerate(as_completed(futures)):
            fu.result()
            if i % 20 == 0:
                el = time.time() - t0
                print(f'{len(done)}/{-(-SIZE // CHUNK)} chunks, {got[0]/el/1e6:.1f} MB/s', flush=True)
    print('DOWNLOAD COMPLETE', flush=True)

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
