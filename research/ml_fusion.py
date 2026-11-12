"""Learned decision layer: fuse classical and neural evidence into a calibrated match probability.

For every query there are two candidate songs: the classical (v2) top song and the neural top
song. Two classifiers estimate P(candidate is the right song | evidence) from the same feature
vector. The more probable candidate is reported, and a split-conformal threshold computed on
held-out calibration negatives turns the probability into a decision with a guaranteed
false-match rate (<= alpha for exchangeable non-library queries).

Features: v2 -log10 p, best score, second score, their ratio, total votes, hashes;
v1-fixed -log10 p; neural best/second similarity and margin; whether the methods agree; query length.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss

from research import stats

FEATURES = ['v2_nlp', 'v2_score', 'v2_second', 'v2_ratio', 'v2_logM', 'v2_loghash', 'v1f_nlp',
            'nn_score', 'nn_second', 'nn_margin', 'agree_v2_nn', 'agree_v1f_v2', 'logL']


def nlp(p):
    return float(min(300.0, -np.log10(max(p, 1e-300))))


def features(spec, c, n):
    """c: classical result for this query (library 4000), n: neural result tuple (track, score, offset, second)."""
    v2, v1f = c['v2'], c['v1f']
    return [nlp(v2['p']), v2['score'], v2['second'], v2['score'] / (v2['second'] + 1), np.log1p(v2['M']),
            np.log1p(v2['n_hash']), nlp(v1f['p']), n[1], n[3], n[1] - n[3], float(n[0] == v2['top']),
            float(v1f['top'] == v2['top']), np.log(spec['L'])]


def make_model(kind):
    if kind == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0)


class Fusion:
    def __init__(self, kind='gbm'):
        self.kind = kind

    def fit(self, X, ok_v2, ok_nn, X_cal, cv2_cal, cnn_cal, label_cal, alpha=0.01, L_cal=None, train_label=None):
        """ok_v2 / ok_nn: whether the classical / neural candidate is the right song (training labels).
        train_label: 1 = library query, 0 = non-library; used to weight both kinds equally (50/50 prior),
        so probabilities are not inflated by the 2:1 positive share of the calibration set."""
        w = None
        if train_label is not None:
            n1, n0 = max(1, (train_label == 1).sum()), max(1, (train_label == 0).sum())
            w = np.where(train_label == 1, 0.5 / n1, 0.5 / n0) * len(train_label)
        self.m_v2 = self._fit(make_model(self.kind), X, ok_v2.astype(int), w)
        self.m_nn = self._fit(make_model(self.kind), X, ok_nn.astype(int), w)
        _, conf_cal = self.predict(X_cal, cv2_cal, cnn_cal, raw=True)
        self.alpha = alpha
        self.threshold = stats.conformal_threshold(conf_cal[label_cal == 0], alpha)
        # Mondrian conformal: one threshold per query length
        self.thresholds = None
        if L_cal is not None:
            self.thresholds = stats.mondrian_thresholds(conf_cal[label_cal == 0], L_cal[label_cal == 0], alpha)
        return self

    @staticmethod
    def _fit(model, X, y, w):
        if w is None:
            return model.fit(X, y)
        if hasattr(model, 'steps'):
            return model.fit(X, y, **{model.steps[-1][0] + '__sample_weight': w})
        return model.fit(X, y, sample_weight=w)

    def predict(self, X, cand_v2, cand_nn, raw=False):
        """Chosen song and its probability (raw=True: log-odds, which do not saturate at 1.0 and so
        give the conformal threshold full resolution)."""
        if raw:
            p2, pn = self.m_v2.decision_function(X), self.m_nn.decision_function(X)
        else:
            p2, pn = self.m_v2.predict_proba(X)[:, 1], self.m_nn.predict_proba(X)[:, 1]
        use_v2 = p2 >= pn
        return np.where(use_v2, cand_v2, cand_nn), np.maximum(p2, pn)

    def decide(self, X, cand_v2, cand_nn, L=None):
        song, conf = self.predict(X, cand_v2, cand_nn, raw=True)
        if self.thresholds is not None and L is not None:
            thr = np.array([self.thresholds.get(l, self.threshold) for l in np.atleast_1d(L)])
        else:
            thr = self.threshold
        return song, conf, conf > thr


def reliability(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.sum() >= 5:
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out
