"""Turn the raw benchmark results into tables (results/summary.json) and figures (results/*.png).

python -m research.analyze
"""
import os, re, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research import stats, augment
from research.bench import WORK, RESULTS, PHIS
from research import ml_fusion

ALPHA = 0.01
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'audfprint (default)': '#d9b8a0', 'audfprint (conformal)': '#a8735a', 'v1 original (score>39)': '#9aa1ae', 'v1 original + Poisson': '#c08a2e', 'v1 fixed + Poisson': '#d4612a',
       'v2 + Poisson': '#8fb0f0', 'v2 + quasi-Poisson': '#2f5fd0', 'v2 + conformal': '#1b3a8a',
       'neural (conformal)': '#2a9d6f', 'hybrid ML (conformal)': '#7b3fbf'}
METHODS = list(COL)
FIG_METHODS = ['audfprint (conformal)', 'v1 original (score>39)', 'v2 + quasi-Poisson', 'neural (conformal)',
               'hybrid ML (conformal)']
PHI_HAT = None


def load():
    specs = pickle.load(open(os.path.join(WORK, 'specs.pkl'), 'rb'))
    cl = pickle.load(open(os.path.join(WORK, 'classical.pkl'), 'rb'))
    nn_path = os.path.join(WORK, 'neural_results.pkl')
    nn = pickle.load(open(nn_path, 'rb')) if os.path.exists(nn_path) else None
    af_path = os.path.join(WORK, 'audfprint_results.pkl')
    global AF
    AF = pickle.load(open(af_path, 'rb')) if os.path.exists(af_path) else None
    mark_duplicates(specs)
    return specs, cl, nn


AF = None
LIB_GROUP = None          # library song id -> duplicate-group key


def _norm_title(t):
    t = re.sub(r'[\(\[].*?[\)\]]', ' ', str(t).lower())
    return re.sub(r'[^a-z0-9]+', '', t)


def mark_duplicates(specs):
    """FMA contains the same music under several track ids (re-uploads, vocal/instrumental versions).
    Tracks with the same artist and the same base title (text in brackets removed) form one group.
    Non-library queries whose group exists in the library are excluded from false-match counts,
    and naming any library track of the right group counts as correct. Uses metadata only."""
    global LIB_GROUP
    from research import data
    tracks, _ = data.cached()
    sp = data.splits()
    md = data.track_metadata()

    def key(name):
        tid = int(os.path.basename(name)[:-4])
        m = md.get(tid, {})
        title = _norm_title(m.get('title', ''))
        return (str(m.get('artist', '')).strip().lower(), title) if title else ('#', tid)

    LIB_GROUP = [key(tracks[i]) for i in sp['library'][:4000]]
    lib_set = set(LIB_GROUP)
    by_artist = {}
    for a, t in lib_set:
        by_artist.setdefault(a, []).append(t)

    def related(k):
        # same artist and one base title contains the other ("To Absent Friends" / "...this Christmas")
        if k in lib_set:
            return True
        a, t = k
        return a != '#' and isinstance(t, str) and len(t) >= 6 and any(
            isinstance(u, str) and len(u) >= 6 and (u in t or t in u) for u in by_artist.get(a, []))
    for sp_ in specs:
        sp_['dup'] = sp_['label'] == 0 and related(key(sp_['track']))
    return sum(x['dup'] for x in specs)


def fpr_ci(ids, specs_by_id, dec, method, n_boot=2000, seed=0):
    """95% interval for the false-match rate, bootstrapping over SONGS (queries from one song are correlated)."""
    neg = [i for i in ids if specs_by_id[i]['label'] == 0 and not specs_by_id[i]['dup'] and method in dec.get(i, {})]
    if not neg:
        return [np.nan, np.nan]
    tracks = {}
    for i in neg:
        tracks.setdefault(specs_by_id[i]['track'], []).append(float(dec[i][method][0]))
    sums = np.array([sum(v) for v in tracks.values()]); cnts = np.array([len(v) for v in tracks.values()])
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(sums), (n_boot, len(sums)))
    rates_ = sums[b].sum(1) / cnts[b].sum(1)
    return [float(np.quantile(rates_, 0.025)), float(np.quantile(rates_, 0.975))]


def is_correct(spec, pred):
    return pred is not None and pred >= 0 and spec['label'] == 1 and LIB_GROUP[pred] == LIB_GROUP[spec['song']]


def decisions(specs, cl, nn, lib=4000):
    """Per query and method: (matched?, predicted song)."""
    by_id = {s['id']: s for s in specs}
    # neural conformal threshold from calibration negatives
    nn_thr = None
    if nn:
        cal = [s for s in specs if s['grid'] == 'conf' and not s['dup'] and s['id'] in nn['results']]
        nn_thr = stats.mondrian_thresholds([nn['results'][s['id']][4000][1] for s in cal], [s['L'] for s in cal], ALPHA)
    global PHI_HAT
    if PHI_HAT is None:
        neg = [s for s in specs if s['label'] == 0 and not s['dup'] and s['grid'] in ('calib', 'conf') and cl.get(s['id'])]
        fpr = {k: np.mean([cl[s['id']][4000]['v2']['p_phi'][k] < ALPHA for s in neg]) for k in range(len(PHIS))}
        PHI_HAT = next((k for k in range(len(PHIS)) if fpr[k] <= ALPHA), len(PHIS) - 1)
    conf_set = [s for s in specs if s['grid'] == 'conf' and not s['dup'] and cl.get(s['id'])]
    v2_thr = stats.mondrian_thresholds([ml_fusion.nlp(cl[s['id']][4000]['v2']['p']) for s in conf_set],
                                       [s['L'] for s in conf_set], ALPHA)
    fusion = fit_fusion(specs, cl, nn) if (nn and lib == 4000) else None
    fus = {}
    if fusion is not None:
        ts = [s for s in specs if s['grid'] in ('main', 'length') and cl.get(s['id']) and s['id'] in nn['results']]
        X = np.array([ml_fusion.features(s, cl[s['id']][4000], nn['results'][s['id']][4000]) for s in ts])
        song, conf, ok = fusion.decide(X, np.array([cl[s['id']][4000]['v2']['top'] for s in ts]),
                                       np.array([nn['results'][s['id']][4000][0] for s in ts]), L=[s['L'] for s in ts])
        fus = {s['id']: (bool(ok[i]), int(song[i])) for i, s in enumerate(ts)}
    af_thr = None
    if AF:
        conf = [s for s in specs if s['grid'] == 'conf' and not s['dup'] and s['id'] in AF['results']]
        af_thr = stats.mondrian_thresholds([AF['results'][s['id']][1] for s in conf], [s['L'] for s in conf], ALPHA)
    out = {}
    for s in specs:
        c = cl.get(s['id'])
        if c is None or s['grid'] == 'seq' or lib not in c:
            continue
        c = c[lib]
        d = {'v1 original (score>39)': (c['v1o']['rule_song'] >= 0, c['v1o']['rule_song']),
             'v1 original + Poisson': (c['v1o']['p'] < ALPHA, c['v1o']['top']),
             'v1 fixed + Poisson': (c['v1f']['p'] < ALPHA, c['v1f']['top']),
             'v2 + Poisson': (c['v2']['p'] < ALPHA, c['v2']['top']),
             'v2 + quasi-Poisson': (c['v2']['p_phi'][PHI_HAT] < ALPHA, c['v2']['top']),
             'v2 + conformal': (ml_fusion.nlp(c['v2']['p']) > v2_thr[s['L']], c['v2']['top'])}
        if AF and lib == 4000 and s['id'] in AF['results']:
            a = AF['results'][s['id']]
            d['audfprint (default)'] = (a[1] >= 5, a[0])
            d['audfprint (conformal)'] = (a[1] > af_thr[s['L']], a[0])
        if nn and s['id'] in nn['results'] and lib in nn['results'][s['id']]:
            n = nn['results'][s['id']][lib]
            d['neural (conformal)'] = (n[1] > nn_thr[s['L']], n[0])
            if s['id'] in fus:
                d['hybrid ML (conformal)'] = fus[s['id']]
        out[s['id']] = d
    return out, nn_thr, fusion


def fit_fusion(specs, cl, nn):
    """Train on the calibration set, take conformal thresholds from the separate conformal set."""
    tr = [s for s in specs if s['grid'] == 'calib' and not s['dup'] and cl.get(s['id']) and s['id'] in nn['results']]
    ca = [s for s in specs if s['grid'] == 'conf' and not s['dup'] and cl.get(s['id']) and s['id'] in nn['results']]

    def mats(ss):
        X = np.array([ml_fusion.features(s, cl[s['id']][4000], nn['results'][s['id']][4000]) for s in ss])
        cv2 = np.array([cl[s['id']][4000]['v2']['top'] for s in ss])
        cnn = np.array([nn['results'][s['id']][4000][0] for s in ss])
        ok2 = np.array([is_correct(s, c) for s, c in zip(ss, cv2)])
        okn = np.array([is_correct(s, c) for s, c in zip(ss, cnn)])
        return X, cv2, cnn, ok2, okn, np.array([s['label'] for s in ss]), np.array([s['L'] for s in ss])

    X, cv2, cnn, ok2, okn, lab, _ = mats(tr)
    Xc, cv2c, cnnc, _, _, labc, Lc = mats(ca)
    # model choice by 5-fold cross-validated log-loss on the calibration set
    from sklearn.model_selection import StratifiedKFold
    best, best_ll = None, np.inf
    for kind in ('logistic', 'gbm'):
        lls = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            f = ml_fusion.Fusion(kind).fit(X[a], ok2[a], okn[a], X[b], cv2[b], cnn[b], lab[b], ALPHA, train_label=lab[a])
            p2 = f.m_v2.predict_proba(X[b])[:, 1]; pn = f.m_nn.predict_proba(X[b])[:, 1]
            conf = np.maximum(p2, pn)
            correct = np.where(p2 >= pn, ok2[b], okn[b]).astype(int)
            lls.append(ml_fusion.log_loss(correct, np.clip(conf, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if np.mean(lls) < best_ll:
            best_ll, best = np.mean(lls), kind
    f = ml_fusion.Fusion(best).fit(X, ok2, okn, Xc, cv2c, cnnc, labc, ALPHA, L_cal=Lc, train_label=lab)
    f.cv_logloss = best_ll
    return f


def rates(ids, specs_by_id, dec, method):
    pos = [i for i in ids if specs_by_id[i]['label'] == 1 and method in dec.get(i, {})]
    neg = [i for i in ids if specs_by_id[i]['label'] == 0 and not specs_by_id[i]['dup'] and method in dec.get(i, {})]
    tpr = np.mean([dec[i][method][0] and is_correct(specs_by_id[i], dec[i][method][1]) for i in pos]) if pos else np.nan
    fpr = np.mean([dec[i][method][0] for i in neg]) if neg else np.nan
    top1 = np.mean([is_correct(specs_by_id[i], dec[i][method][1]) for i in pos]) if pos else np.nan
    return float(tpr), float(fpr), float(top1), len(pos), len(neg)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    specs, cl, nn = load()
    S = {s['id']: s for s in specs}
    dec, nn_thr, fusion = decisions(specs, cl, nn)
    summary = {'duplicates_excluded': int(sum(s['dup'] for s in specs)), 'alpha': ALPHA, 'neural_threshold': {str(k): float(v) for k, v in (nn_thr or {}).items()}}
    if fusion is not None:
        summary['fusion_model'] = fusion.kind
        summary['fusion_cv_logloss'] = float(fusion.cv_logloss)
        summary['fusion_threshold'] = {str(k): float(v) for k, v in fusion.thresholds.items()}

    # ---- main grid (5 s) and length grid
    main_tab, len_tab = {}, {}
    for cond in augment.CONDITIONS:
        ids = [s['id'] for s in specs if s['grid'] == 'main' and s['cond'] == cond]
        main_tab[cond] = {m: rates(ids, S, dec, m) for m in METHODS}
    for cond in ['clean', 'pink5', 'mic']:
        for L in [1.0, 2.0, 3.0, 5.0, 10.0]:
            grid = 'main' if L == 5.0 else 'length'
            ids = [s['id'] for s in specs if s['grid'] == grid and s['cond'] == cond and s['L'] == L]
            len_tab[f'{cond}/{L:g}'] = {m: rates(ids, S, dec, m) for m in METHODS}
    allids = [s['id'] for s in specs if s['grid'] in ('main', 'length')]
    summary['overall'] = {m: rates(allids, S, dec, m) for m in METHODS}
    summary['overall_fpr_ci'] = {m: fpr_ci(allids, S, dec, m) for m in METHODS}
    summary['main'] = main_tab
    summary['length'] = len_tab

    # ---- scaling with library size
    scale = {}
    for n in [100, 300, 1000, 2000, 4000]:
        d_n, _, _ = decisions(specs, cl, nn, lib=n) if n != 4000 else (dec, None, None)
        for cond in ['clean', 'mic']:
            ids = [s['id'] for s in specs if s['grid'] == 'scale' and s['cond'] == cond]
            scale[f'{n}/{cond}'] = {m: rates(ids, S, d_n, m) for m in METHODS if m.startswith(('v1 original (', 'v2 + quasi', 'v2 + Poisson', 'neural'))}
        ids_both = [s['id'] for s in specs if s['grid'] == 'scale']
        scale[f'{n}/fpr_ci'] = {m: fpr_ci(ids_both, S, d_n, m) for m in METHODS if m.startswith(('v1 original (', 'v2 + quasi', 'v2 + Poisson', 'neural'))}
    summary['scale'] = scale
    # latency (ms per query, 4000 songs, matching only)
    lat = {'v1 original (score>39)': [], 'v1 original + Poisson': [], 'v2 + Poisson': []}
    for s in specs:
        c = cl.get(s['id'])
        if c and s['grid'] == 'main' and 4000 in c:
            lat['v1 original (score>39)'].append(c[4000]['v1o']['rule_ms'])
            lat['v1 original + Poisson'].append(c[4000]['v1o']['ms'])
            lat['v2 + Poisson'].append(c[4000]['v2']['ms'])
    summary['latency_ms_median'] = {k: float(np.median(v)) for k, v in lat.items() if v}
    if nn:
        summary['latency_ms_median']['neural (search)'] = nn['search_ms']

    # ---- calibration of the Poisson model on non-library queries
    alphas = np.logspace(-4, -0.3, 25)
    negs = [s for s in specs if s['label'] == 0 and not s['dup'] and s['grid'] in ('main', 'length') and cl.get(s['id'])]
    calnegs = [s for s in specs if s['label'] == 0 and not s['dup'] and s['grid'] in ('calib', 'conf') and cl.get(s['id'])]
    calib = {}
    for name, get in [('v2 anchors (Poisson)', lambda c: c['v2']['p']), ('v2 pairs (Poisson)', lambda c: c['v2']['p_pairs']),
                      ('v1 fixed (Poisson)', lambda c: c['v1f']['p'])]:
        ps = np.array([get(cl[s['id']][4000]) for s in negs])
        calib[name] = [float(np.mean(ps < a)) for a in alphas]
    # quasi-Poisson: phi chosen on calibration negatives so that FPR(alpha=1%) <= 1%
    phi_fpr = {phi: float(np.mean([cl[s['id']][4000]['v2']['p_phi'][k] < ALPHA for s in calnegs])) for k, phi in enumerate(PHIS)}
    phi_hat = next((phi for phi in PHIS if phi_fpr[phi] <= ALPHA), PHIS[-1])
    k_hat = PHIS.index(phi_hat)
    ps = np.array([cl[s['id']][4000]['v2']['p_phi'][k_hat] for s in negs])
    calib[f'v2 quasi-Poisson (phi={phi_hat})'] = [float(np.mean(ps < a)) for a in alphas]
    summary['calibration'] = {'alphas': alphas.tolist(), 'curves': calib, 'phi_fpr_on_calib': phi_fpr, 'phi_hat': phi_hat,
                              'v1f_fpr_at_alpha_by_phi': {phi: float(np.mean([cl[s['id']][4000]['v1f']['p_phi'][k] < ALPHA for s in calnegs]))
                                                           for k, phi in enumerate(PHIS)}}
    pos_main = [s for s in specs if s['label'] == 1 and s['grid'] in ('main', 'length') and cl.get(s['id'])]
    summary['phi_effect'] = {
        'tpr_poisson': float(np.mean([(cl[s['id']][4000]['v2']['p'] < ALPHA) and is_correct(s, cl[s['id']][4000]['v2']['top']) for s in pos_main])),
        'tpr_quasi': float(np.mean([(cl[s['id']][4000]['v2']['p_phi'][k_hat] < ALPHA) and is_correct(s, cl[s['id']][4000]['v2']['top']) for s in pos_main])),
        'fpr_poisson': float(np.mean([cl[s['id']][4000]['v2']['p'] < ALPHA for s in negs])),
        'fpr_quasi': float(np.mean(ps < ALPHA))}

    # ---- sequential decisions
    seq = {}
    for cond in ['clean', 'pink5', 'mic']:
        ss = [s for s in specs if s['grid'] == 'seq' and s['cond'] == cond and not s['dup'] and cl.get(s['id'])]
        row = {}
        for mode in ['naive', 'bonferroni', 'single test at 10 s']:
            tp, fp, times, npos, nneg = 0, 0, [], 0, 0
            for s in ss:
                ps = cl[s['id']]['seq']
                pv = [x[3][PHI_HAT] for x in ps]
                if mode == 'single test at 10 s':
                    ok, t = (pv[-1] < ALPHA, 9) if pv[-1] < ALPHA else (False, None)
                else:
                    ok, t = stats.sequential_decision(pv, ALPHA, mode)
                if s['label'] == 1:
                    npos += 1
                    if ok and is_correct(s, ps[t][1]):
                        tp += 1; times.append(t + 1)
                else:
                    nneg += 1; fp += ok
            row[mode] = {'tpr': tp / npos, 'fpr': fp / nneg, 'mean_time_s': float(np.mean(times)) if times else None,
                         'times': times}
        seq[cond] = row
    summary['sequential'] = seq

    # ---- fusion reliability on test queries
    if fusion is not None:
        test = [s for s in specs if s['grid'] in ('main', 'length') and cl.get(s['id']) and s['id'] in nn['results']]
        X = np.array([ml_fusion.features(s, cl[s['id']][4000], nn['results'][s['id']][4000]) for s in test])
        song, conf = fusion.predict(X, np.array([cl[s['id']][4000]['v2']['top'] for s in test]),
                                    np.array([nn['results'][s['id']][4000][0] for s in test]))
        keep = np.array([not s['dup'] for s in test])
        correct = np.array([is_correct(s, song[i]) for i, s in enumerate(test)]).astype(int)
        conf, correct = conf[keep], correct[keep]
        summary['fusion_reliability'] = ml_fusion.reliability(conf, correct)
        summary['fusion_brier'] = float(ml_fusion.brier_score_loss(correct, conf))

    json.dump(summary, open(os.path.join(RESULTS, 'summary.json'), 'w'), indent=1, default=float)
    figures(summary, nn)
    print_tables(summary)


def figures(sm, nn):
    # 1. robustness (5 s queries)
    conds = ['clean', 'pink20', 'pink10', 'pink5', 'pink0', 'reverb', 'mic', 'speed+1%', 'speed-2%', 'speed+5%']
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.16
    ms = [m for m in FIG_METHODS if not np.isnan(sm['main']['clean'][m][0])]
    for k, m in enumerate(ms):
        ax.bar(np.arange(len(conds)) + (k - len(ms) / 2 + 0.5) * w, [sm['main'][c][m][0] for c in conds], w, color=COL[m], label=m)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(['clean', 'noise\n20 dB', '10 dB', '5 dB', '0 dB', 'reverb', 'phone\nmic',
                                                          'speed\n+1%', '-2%', '+5%'])
    ax.set_ylabel('correct matches (TPR)'); ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, fontsize=7, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.22))
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_robustness.png'), dpi=250); plt.close(fig)

    # 2. query length
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, cond, title in zip(axs, ['clean', 'pink5', 'mic'], ['clean', 'noise 5 dB', 'phone mic']):
        for m in ms:
            ax.plot([1, 2, 3, 5, 10], [sm['length'][f'{cond}/{L:g}'][m][0] for L in [1.0, 2.0, 3.0, 5.0, 10.0]], 'o-', ms=3, color=COL[m], label=m)
        ax.set_title(title, fontsize=9); ax.set_xlabel('query length (s)'); ax.set_xscale('log'); ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels(['1', '2', '3', '5', '10'])
    axs[0].set_ylabel('TPR'); axs[0].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_length.png'), dpi=250); plt.close(fig)

    # 3. library-size scaling
    sizes = [100, 300, 1000, 2000, 4000]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for m in sm['scale']['100/clean']:
        if np.isnan(sm['scale']['100/clean'][m][0]):
            continue
        axs[0].plot(sizes, [sm['scale'][f'{n}/mic'][m][0] for n in sizes], 'o-', ms=3, color=COL[m], label=m)
        fp = [max(1e-4, (sm['scale'][f'{n}/clean'][m][1] + sm['scale'][f'{n}/mic'][m][1]) / 2) for n in sizes]
        lo = [max(1e-4, sm['scale'][f'{n}/fpr_ci'][m][0]) for n in sizes]
        hi = [max(1e-4, sm['scale'][f'{n}/fpr_ci'][m][1]) for n in sizes]
        axs[1].plot(sizes, fp, 'o-', ms=3, color=COL[m])
        axs[1].fill_between(sizes, lo, hi, color=COL[m], alpha=0.12, lw=0)
    axs[0].set_title('correct matches, phone-mic queries', fontsize=9); axs[0].set_xscale('log'); axs[0].set_xlabel('songs in library'); axs[0].set_ylim(0, 1.02)
    axs[1].axhline(ALPHA, color='k', lw=0.6, ls='--'); axs[1].text(110, ALPHA * 1.15, 'target 1%', fontsize=7)
    axs[1].set_title('false-match rate (non-library queries)', fontsize=9); axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('songs in library')
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_scaling.png'), dpi=250); plt.close(fig)

    # 4. calibration
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    a = np.array(sm['calibration']['alphas'])
    for name, cur in sm['calibration']['curves'].items():
        ax.plot(a, np.maximum(cur, 1e-4), label=name, lw=1.2)
    ax.plot(a, a, 'k--', lw=0.7, label='perfect calibration')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('nominal false-match level α'); ax.set_ylabel('observed false-match rate')
    ax.set_title('is the p-value honest? (non-library queries)', fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_calibration.png'), dpi=250); plt.close(fig)

    # 5. sequential: time to decision (positives)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for cond, ls in zip(['clean', 'pink5', 'mic'], ['-', '--', ':']):
        for mode, colr in [('naive', '#c08a2e'), ('bonferroni', '#2f5fd0')]:
            t = np.array(sm['sequential'][cond][mode]['times'])
            npos = 300
            ax.step(range(1, 11), [np.sum(t <= k) / npos for k in range(1, 11)], where='post', ls=ls, color=colr,
                    label=f'{cond}, {mode}', lw=1.1)
    ax.set_xlabel('seconds of audio heard'); ax.set_ylabel('songs identified (cumulative)'); ax.set_ylim(0, 1.02)
    ax.set_title('time to identification (10 s queries)', fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sequential.png'), dpi=250); plt.close(fig)

    # 6. neural training loss + fusion reliability
    if nn and os.path.exists(os.path.join(WORK, 'neural_train.json')):
        tr = json.load(open(os.path.join(WORK, 'neural_train.json')))
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
        loss = np.array(tr['loss'])
        axs[0].plot(np.convolve(loss, np.ones(50) / 50, mode='valid'), color='#2a9d6f')
        axs[0].set_xlabel('training step'); axs[0].set_ylabel('NT-Xent loss'); axs[0].set_title('neural fingerprint training', fontsize=9)
        if 'fusion_reliability' in sm:
            r = np.array(sm['fusion_reliability'])
            axs[1].plot([0, 1], [0, 1], 'k--', lw=0.7)
            axs[1].plot(r[:, 0], r[:, 1], 'o-', color='#7b3fbf', ms=3)
            axs[1].set_xlabel('predicted probability of a correct match'); axs[1].set_ylabel('observed frequency')
            axs[1].set_title(f"fusion calibration (Brier {sm['fusion_brier']:.3f})", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_neural.png'), dpi=250); plt.close(fig)


def print_tables(sm):
    print('OVERALL (main+length grids): TPR / FPR / top1')
    for m, r in sm['overall'].items():
        ci = sm['overall_fpr_ci'][m]
        print(f'  {m:28s} {r[0]:.3f} {r[1]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {r[2]:.3f}  (n+={r[3]}, n-={r[4]})')
    print('MAIN (5 s) TPR per condition')
    for c, row in sm['main'].items():
        print(f'  {c:9s}', ' '.join(f'{row[m][0]:.2f}/{row[m][1]:.3f}' for m in METHODS))
    print('LATENCY', sm['latency_ms_median'])
    print('CALIBRATION phi', sm['calibration']['phi_fpr_on_calib'], 'phi_hat', sm['calibration']['phi_hat'], sm['phi_effect'])
    for cond, row in sm['sequential'].items():
        print('SEQ', cond, {k: (round(v['tpr'], 3), round(v['fpr'], 4), v['mean_time_s']) for k, v in row.items()})
    for k, row in sm['scale'].items():
        if k.endswith('fpr_ci'):
            print('SCALE', k, {m: (round(v[0], 4), round(v[1], 4)) for m, v in row.items()})
        else:
            print('SCALE', k, {m: (round(v[0], 3), round(v[1], 4)) for m, v in row.items()})


if __name__ == '__main__':
    main()
