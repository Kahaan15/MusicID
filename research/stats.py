"""Statistics for the match decision.

Null hypothesis H0: the query is not from song s, so its votes for s land on time offsets
"at random". With M_s votes spread over K_s possible offsets, the count in one offset
window of width w is approximately Poisson(w * M_s / K_s). The best window must beat the
maximum over K_s windows and over every song in the library, so the family-wise p-value of
a top score c is

    p(c) = 1 - prod_s P(max_s < c)  ~=  1 - exp( - sum_s K_s * P(Poisson(w*lambda_s) >= c) )

A match is declared when p(c) < alpha. The threshold therefore grows automatically with
library size, number of votes and query length, unlike a fixed "score > 39".
"""
import numpy as np
from scipy.stats import poisson
from scipy.special import pdtrc      # pdtrc(k, m) = P(Poisson(m) > k), much faster than poisson.sf


def family_pvalue(votes, K_song, K_query, radius=1, phi=1.0):
    """p-value of the best song's score under H0.

    votes   : dict from index.offset_votes (song, score, M)
    K_song  : array of song lengths (time units) for the song ids in votes['song']
    K_query : query length (time units)
    phi     : over-dispersion factor (1 = pure Poisson). Counts are divided by phi
              (a quasi-Poisson correction estimated on held-out negatives).
    Returns (p_value, top_song, top_score, second_score).
    """
    s, score, M = votes['song'], votes['score'].astype(np.float64), votes['M'].astype(np.float64)
    if len(s) == 0:
        return 1.0, -1, 0, 0
    order = np.argsort(-score)
    top = order[0]
    c = score[top]
    second = score[order[1]] if len(order) > 1 else 0.0
    K = K_song.astype(np.float64) + K_query
    w = 2 * radius + 1
    lam = w * M / K
    # quasi-Poisson: score/phi ~ Poisson(lam/phi)
    tail = pdtrc(np.ceil(c / phi) - 1, lam / phi)
    log_none = -np.sum(K * tail)
    return float(-np.expm1(log_none)), int(s[top]), int(c), int(second)


def score_threshold(M, K_song, K_query, alpha, radius=1, n_songs=1):
    """Smallest score c with family-wise false-match probability <= alpha (for plotting).

    Assumes n_songs songs each with M votes spread over K_song + K_query offsets.
    """
    K = K_song + K_query
    lam = (2 * radius + 1) * M / K
    c = np.arange(1, 2000)
    fw = -np.expm1(-n_songs * K * poisson.sf(c - 1, lam))
    return int(c[np.argmax(fw <= alpha)])


def conformal_threshold(null_scores, alpha):
    """Split-conformal threshold on a 'higher = more match-like' score.

    If test negatives are exchangeable with the calibration negatives, declaring a match
    when score > threshold gives P(false match) <= alpha (finite-sample guarantee).
    """
    s = np.sort(np.asarray(null_scores))
    n = len(s)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else s[k - 1]


def mondrian_thresholds(null_scores, groups, alpha):
    """Conformal threshold per group (e.g. per query length): the <= alpha guarantee holds inside each group."""
    null_scores, groups = np.asarray(null_scores), np.asarray(groups)
    return {g: conformal_threshold(null_scores[groups == g], alpha) for g in np.unique(groups)}


def estimate_phi(null_neglogp_fn, phis, alpha, target):
    """Pick the smallest over-dispersion factor whose empirical FPR on calibration negatives <= target."""
    for phi in phis:
        if null_neglogp_fn(phi) <= target:
            return phi
    return phis[-1]


def sequential_decision(pvals, alpha, mode='bonferroni'):
    """Decide as audio arrives. pvals[t] = p-value after t+1 looks.

    naive      : stop at the first look with p < alpha  (inflates the false-match rate: "peeking")
    bonferroni : alpha is split over the T looks (alpha / T each)  -> overall error <= alpha
    Returns (decided_match, look_index or None).
    """
    T = len(pvals)
    a = alpha if mode == 'naive' else alpha / T
    for t, p in enumerate(pvals):
        if p < a:
            return True, t
    return False, None
