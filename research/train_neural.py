"""Train the neural fingerprint on the training pool (songs disjoint from the evaluation library).

python -m research.train_neural [steps] [batch]
"""
import os, sys, json, time
import numpy as np
from research import data, neural
from research.bench import WORK


def main(steps=2500, batch=192):
    os.makedirs(WORK, exist_ok=True)
    _, mm = data.cached()
    sp = data.splits()
    log_f = open(os.path.join(WORK, 'neural_train.log'), 'w')

    def log(s):
        print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

    log(f'training tracks: {len(sp["train"])}, steps {steps}, batch {batch}')
    t0 = time.time()
    model, hist = neural.train(mm, sp['train'], steps=steps, batch=batch, log=log,
                               out=os.path.join(WORK, "neural.pt"), threads=8)
    json.dump({'loss': hist, 'seconds': time.time() - t0, 'steps': steps, 'batch': batch,
               'train_tracks': int(len(sp['train']))}, open(os.path.join(WORK, 'neural_train.json'), 'w'))
    log('TRAINING DONE')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 2500, int(a[1]) if len(a) > 1 else 192)
