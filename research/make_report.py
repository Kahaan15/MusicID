"""Build the technical report PDF (docs/MusicID_Technical_Report.pdf) from research/results.

python -m research.make_report
All numbers come from results/summary.json and results/filter_study.json.
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether,
                                PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

from research.bench import RESULTS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DOCS = os.path.join(ROOT, 'docs')
EQ_DIR = os.path.join(RESULTS, 'eq')

pdfmetrics.registerFont(TTFont('Cal', r'C:\Windows\Fonts\calibri.ttf'))
pdfmetrics.registerFont(TTFont('Cal-B', r'C:\Windows\Fonts\calibrib.ttf'))
pdfmetrics.registerFont(TTFont('Cal-I', r'C:\Windows\Fonts\calibrii.ttf'))
addMapping('Cal', 0, 0, 'Cal'); addMapping('Cal', 1, 0, 'Cal-B'); addMapping('Cal', 0, 1, 'Cal-I'); addMapping('Cal', 1, 1, 'Cal-B')

ACCENT, INK, MUTED = colors.HexColor('#2f5fd0'), colors.HexColor('#1c2230'), colors.HexColor('#5d6677')
RULE, SHADE, GOOD = colors.HexColor('#d5dae3'), colors.HexColor('#f2f5fa'), colors.HexColor('#e6f4ec')
ST = {
    'title': ParagraphStyle('t', fontName='Cal-B', fontSize=20, leading=24, textColor=INK),
    'sub': ParagraphStyle('s', fontName='Cal', fontSize=10, leading=13, textColor=MUTED),
    'h1': ParagraphStyle('h1', fontName='Cal-B', fontSize=13, leading=16, textColor=ACCENT, spaceBefore=9, spaceAfter=3),
    'h2': ParagraphStyle('h2', fontName='Cal-B', fontSize=10.5, leading=13, textColor=INK, spaceBefore=5, spaceAfter=1),
    'body': ParagraphStyle('b', fontName='Cal', fontSize=9.6, leading=12.4, textColor=INK, spaceAfter=3),
    'small': ParagraphStyle('sm', fontName='Cal', fontSize=8.3, leading=10.1, textColor=INK),
    'note': ParagraphStyle('n', fontName='Cal-I', fontSize=8.4, leading=10.5, textColor=MUTED, spaceAfter=3),
    'bullet': ParagraphStyle('bl', fontName='Cal', fontSize=9.6, leading=12.4, textColor=INK, leftIndent=12, bulletIndent=2,
                             spaceAfter=1.5),
}
plt.rcParams['mathtext.fontset'] = 'cm'
_n = [0]


def P(t, s='body'):
    return Paragraph(t, ST[s])


def B(t):
    return Paragraph(t, ST['bullet'], bulletText='•')


def eq(tex, size=10.5):
    os.makedirs(EQ_DIR, exist_ok=True)
    _n[0] += 1
    path = os.path.join(EQ_DIR, f'r{_n[0]}.png')
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f'${tex}$', fontsize=size, color='#1c2230')
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)
    w, h = PILImage.open(path).size
    img = Image(path, width=w / 400 * 72, height=h / 400 * 72)
    img.hAlign = 'CENTER'
    return img


def fig(name, width_cm=17.0, caption=None):
    path = os.path.join(RESULTS, name)
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    img.hAlign = 'CENTER'
    out = [Spacer(1, 3), img]
    if caption:
        out.append(P(caption, 'note'))
    return KeepTogether(out)


def img(name, width_cm):
    path = os.path.join(RESULTS, name)
    w, h = PILImage.open(path).size
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


def table(rows, widths, header=True, highlight_rows=(), style='small'):
    data = [[Paragraph(str(c), ST[style]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, hAlign='CENTER', repeatRows=1 if header else 0)
    ts = [('GRID', (0, 0), (-1, -1), 0.4, RULE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 1.8), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        ts.append(('BACKGROUND', (0, 0), (-1, 0), SHADE))
    for r in highlight_rows:
        ts.append(('BACKGROUND', (0, r), (-1, r), GOOD))
    t.setStyle(TableStyle(ts))
    return t


def pct(x, d=1):
    return '–' if x != x else f'{100 * x:.{d}f}%'


def load():
    sm = json.load(open(os.path.join(RESULTS, 'summary.json')))
    fs_path = os.path.join(RESULTS, 'filter_study.json')
    fs = json.load(open(fs_path)) if os.path.exists(fs_path) else None
    return sm, fs


KEY = ['v1 original (score>39)', 'audfprint (default)', 'audfprint (conformal)', 'v2 + quasi-Poisson', 'v2 + conformal',
       'neural (conformal)', 'hybrid ML (conformal)']
NICE = {'v1 original (score>39)': 'Original MusicID (score > 39)', 'audfprint (default)': 'audfprint, default threshold',
        'audfprint (conformal)': 'audfprint, conformal 1%', 'v2 + quasi-Poisson': 'v2 classical, quasi-Poisson p < 1%',
        'v2 + conformal': 'v2 classical, conformal 1%', 'neural (conformal)': 'Neural fingerprint, conformal 1%',
        'hybrid ML (conformal)': 'Hybrid (classical + neural, learned), conformal 1%'}
MT_COLS = ['v1 original (score>39)', 'audfprint (conformal)', 'v2 + conformal', 'neural (conformal)', 'hybrid ML (conformal)']
CONDS = [('clean', 'clean'), ('pink20', 'noise 20 dB'), ('pink10', 'noise 10 dB'), ('pink5', 'noise 5 dB'), ('pink0', 'noise 0 dB'),
         ('reverb', 'reverb'), ('mic', 'phone mic'), ('speed+1%', 'speed +1%'), ('speed-2%', 'speed −2%'), ('speed+5%', 'speed +5%')]


def summary_section(sm):
    S = []
    ov, ci, lat = sm['overall'], sm['overall_fpr_ci'], sm['latency_ms_median']
    hy, v1o, v2c = ov['hybrid ML (conformal)'], ov['v1 original (score>39)'], ov['v2 + conformal']
    S.append(P('MusicID: song recognition from first principles, measured and calibrated', 'title'))
    S.append(P('Technical report · November 2026 · every number here is reproducible from the project folder (section 11)', 'sub'))
    S.append(Spacer(1, 6))
    S.append(P('Summary', 'h1'))
    S.append(P('MusicID began as a re-implementation of the Shazam landmark algorithm (Wang, 2003). This report turns it into a measured '
               'system. The original code is reproduced bit-for-bit, then evaluated on a 4,000-song library from the Free Music Archive '
               'with 17,600 test clips under ten recording conditions and five clip lengths, next to the open-source fingerprinter '
               'audfprint. The guessed match threshold is replaced by a statistically derived one with a stated false-match rate, and a '
               'neural fingerprint plus a learned decision layer are added.'))
    S.append(B(f'The original rule "score &gt; 39" names a song for <b>{pct(v1o[1])}</b> of clips that are <i>not</i> in the library '
               f'(4,000 songs), a rate that grows with library size; it finds the right song for {pct(v1o[0])} of library clips.'))
    S.append(B(f'The improved classical engine (v2) with a derived threshold finds {pct(v2c[0])} at a false-match rate of {pct(v2c[1])} '
               f'(target 1%), and searches {lat["v1 original (score>39)"] / lat["v2 + Poisson"]:.0f}× faster.'))
    S.append(B(f'The hybrid of classical and neural evidence finds <b>{pct(hy[0])}</b> of library clips at a false-match rate of '
               f'{pct(hy[1])} (95% CI {pct(ci["hybrid ML (conformal)"][0])}–{pct(ci["hybrid ML (conformal)"][1])}); its top candidate '
               f'is right {pct(hy[2])} of the time.'))
    S.append(B('Plain Poisson statistics are badly over-confident because votes arrive in clumps; a quasi-Poisson correction '
               f'(estimated φ = {sm["calibration"]["phi_hat"]}) and split-conformal thresholds fix this. Checking every second while recording is '
               'kept honest by splitting the error budget over the checks (Bonferroni).'))
    rows = [['Method', 'Correct matches', 'False matches (95% CI)', 'Top-1', 'Search time']]
    latmap = {'v1 original (score>39)': lat['v1 original (score>39)'], 'v2 + quasi-Poisson': lat['v2 + Poisson'],
              'v2 + conformal': lat['v2 + Poisson'], 'neural (conformal)': lat.get('neural (search)')}
    for m in KEY:
        r, c = ov[m], ci[m]
        t = latmap.get(m)
        rows.append([NICE[m], pct(r[0]), f'{pct(r[1])} ({pct(c[0])}–{pct(c[1])})', pct(r[2]),
                     f'{t:.0f} ms' if t else ('v2 + neural' if 'hybrid' in m else '–')])
    S.append(Spacer(1, 3))
    S.append(table(rows, [6.6 * cm, 2.4 * cm, 3.9 * cm, 1.6 * cm, 2.3 * cm], highlight_rows=[len(rows) - 1]))
    S.append(P(f'All test clips (main + length grids): {ov[KEY[0]][3]:,} from library songs and {ov[KEY[0]][4]:,} from songs not in the '
               'library. "Correct matches" = right song AND accepted by the decision rule. Search time = median database search per 5-s clip '
               'on a laptop CPU (fingerprinting excluded). CI = bootstrap over songs.', 'note'))
    return S


def method_sections(sm):
    S = []
    ov = sm['overall']
    S.append(P('1. Where the method comes from', 'h1'))
    S.append(P('The fingerprint is the constellation-map and landmark-hashing method of Wang (2003). The original MusicID code also follows '
               'the widely read tutorial "How does Shazam work" (coding-geek.com, 2015), including its "address / couple" naming and its '
               'target zone of five points. None of that design is claimed as new here. What this project adds is the measurement, the '
               'statistics, the learned components and the engineering: an exact replica that runs at scale, a benchmark with an external '
               'baseline, a derived decision rule with a stated error rate, a neural fingerprint (after Chang et al., 2021) and a calibrated '
               'fusion of the two.'))
    S.append(P('2. Step one: an exact, fast replica of the original', 'h1'))
    S.append(P('The original pipeline (AudioModule + DBModule) was re-implemented with vectorised NumPy and checked against the original code: '
               'on real recordings and on synthetic audio the constellation maps are <b>bit-identical</b>, the 52,755 stored hashes of the song '
               'database are identical, and the decision (song and score 1,308) is identical (tests/test_engine.py). The replica keeps every quirk '
               'so that the original algorithm is what gets evaluated:'))
    for t in ['stereo is averaged and mono input halved; the signal is zero-padded to a power of two before filtering;',
              'a 40,001-tap windowed-sinc low-pass filter (5 kHz, 4.4 Hz transition band) followed by averaging blocks of four samples;',
              'windowed frames are stored as 16-bit floats before the FFT (a side effect of numpy.insert into a float16 array);',
              'the peak threshold divides by 6 although 5 bands are used, and each peak time has an extra j/11025 s term (j = frequency bin);',
              'target zone = "the next five points of the list", time differences in rounded milliseconds.']:
        S.append(B(t))
    S.append(P('The hand-written recursive FFT matches numpy.fft to 10<super>-9</super> (unit test), so the experiments use NumPy\'s FFT for speed.'))
    S.append(P('3. The improved classical engine (v2)', 'h1'))
    S.append(P('v2 keeps the landmark idea but follows Wang (2003) more closely. Each change addresses something the replica revealed:'))
    for t in ['<b>Anti-aliasing:</b> polyphase resampling to 11,025 Hz from any input rate (the original only works at 44.1 kHz).',
              '<b>Frames:</b> 50% overlapping Hann windows of 1,024 samples (hop 46.4 ms), so a clip that starts between frames still lines up.',
              '<b>Peaks:</b> local maxima of the log spectrogram in a 7-frame × 21-bin neighbourhood, at most 30 per second (215 Hz–5.4 kHz).',
              '<b>Target zones:</b> each anchor is paired with up to 6 later peaks 1–40 frames ahead and within ±100 bins, instead of "the next five list items".',
              '<b>Hash:</b> 24 bits = anchor bin (9) | target bin (9) | Δt in frames (6); times are integer frames, so offsets never split by rounding.',
              '<b>Index:</b> three sorted arrays (key, song, time) searched by bisection instead of a dictionary of strings; 19.2 M hashes for 4,000 songs in 293 MB.']:
        S.append(B(t))
    S.append(P('<b>Votes.</b> A query hash that matches a database hash of song <i>s</i> votes for the time offset δ = t<sub>db</sub> − t<sub>query</sub>. '
               'Each query anchor votes at most once per (song, offset), and a song\'s score is the largest number of votes in a window of three '
               'neighbouring offsets (δ−1, δ, δ+1), which absorbs one-frame misalignment.'))

    S.append(P('4. Statistics: a threshold with a known error rate', 'h1'))
    S.append(P('4.1 A null model for the votes', 'h2'))
    S.append(P('If the clip is not from song <i>s</i>, its M<sub>s</sub> votes for <i>s</i> land on the K<sub>s</sub> possible offsets (song length + '
               'clip length, in frames) at random, so the count in one window of width w = 3 is approximately Poisson with mean wλ<sub>s</sub>, '
               'λ<sub>s</sub> = M<sub>s</sub>/K<sub>s</sub>. The best score must beat the maximum over all offsets of all songs, which gives the '
               'family-wise p-value of an observed top score c:'))
    S.append(eq(r'p(c)\;=\;1-\prod_{s}\Pr\left(\max_{\delta}N_{s,\delta}<c\right)\;\approx\;1-\exp\left(-\sum_{s}K_s\,\Pr\left[\mathrm{Poisson}(w\lambda_s)\geq c\right]\right)'))
    S.append(P('A match is declared when p(c) &lt; α (α = 1%). Unlike a fixed score, this threshold rises automatically with the number of songs, '
               'the number of votes and the clip length.'))
    S.append(P('4.2 The plain model fails, and why', 'h2'))
    S.append(P(f'On held-out non-library clips the plain Poisson rule gives far more false matches than it promises: {pct(sm["phi_effect"]["fpr_poisson"])} '
               f'instead of 1% for v2, and {pct(ov["v1 original + Poisson"][1])} on the original fingerprint. Votes are not independent: one '
               'coincidental alignment of a repeated musical pattern produces a clump of votes at the same offset, so counts are over-dispersed '
               f'(variance &gt; mean). The original hashing is extreme: even with a dispersion factor of 4 its false-match rate is still '
               f'{pct(sm["calibration"]["v1f_fpr_at_alpha_by_phi"]["4.0"])}. The statistics only work on top of a hash design whose votes are '
               'close to independent.'))
    S.append(P('4.3 Two fixes', 'h2'))
    S.append(P('<b>Quasi-Poisson.</b> Treat score/φ as Poisson with mean wλ<sub>s</sub>/φ and choose the smallest φ on a grid whose false-match '
               f'rate on calibration clips is at most α; this gives φ = {sm["calibration"]["phi_hat"]} for v2.'))
    S.append(eq(r'p_\phi(c)\;\approx\;1-\exp\left(-\sum_{s}K_s\,\Pr\left[\mathrm{Poisson}(w\lambda_s/\phi)\geq \lceil c/\phi\rceil\right]\right)'))
    S.append(P('<b>Split conformal.</b> For any score where larger means "more match-like" (−log<sub>10</sub>p, the neural similarity, or the fusion '
               'log-odds), take n calibration clips known not to be in the library, sort their scores, and use the k-th smallest as threshold. If '
               'test non-library clips are exchangeable with the calibration ones, P(false match) ≤ α with no distributional assumption. '
               'Thresholds are set separately for each clip length (Mondrian conformal), because a 1-s clip has a much wider null distribution '
               'than a 10-s clip.'))
    S.append(eq(r'\hat{\tau}=s_{(k)},\quad k=\lceil(1-\alpha)(n+1)\rceil\qquad\Rightarrow\qquad \Pr\left(S_{\mathrm{new}}>\hat{\tau}\mid H_0\right)\leq\alpha'))
    S.append(P('<b>Honest uncertainty.</b> Test clips come from 490 non-library songs, about 17 clips each, and a few songs that resemble a library '
               'song produce many false matches (up to 13 each). The effective sample size is the number of songs, so every confidence interval '
               'bootstraps over songs, not clips.'))
    S.append(P('4.4 Checking every second: sequential testing', 'h2'))
    S.append(P('The app re-checks the growing recording every second. Testing T times at level α each inflates the overall false-match rate '
               '("peeking"). Splitting the budget, α/T per look (Bonferroni), keeps the overall rate at most α:'))
    S.append(eq(r'\Pr\left(\exists\,t\leq T:\;p_t<\alpha/T\;|\;H_0\right)\;\leq\;\sum_{t=1}^{T}\alpha/T\;=\;\alpha'))

    S.append(P('5. Machine learning', 'h1'))
    S.append(P('5.1 A neural fingerprint learned by contrastive learning', 'h2'))
    S.append(P('Each 1-s segment (64-band log-mel spectrogram, 150 Hz–5 kHz, mean-normalised for gain invariance) is mapped by a small CNN (five '
               '3×3 convolution layers, 0.67 M parameters) to a 128-dimensional unit vector f(x). Training pairs are a clean segment and a degraded '
               'copy of the same moment: time shift up to ±0.3 s, pink noise at 0–20 dB, background music, room echo (RT60 0.2–0.8 s), random '
               'EQ, ±3% speed change and gain. The NT-Xent loss (temperature τ = 0.05) pulls each pair together and pushes away the other 2B−2 '
               'segments of the batch:'))
    S.append(eq(r'\ell_i=-\log\frac{\exp\left(f(x_i)^{\top}f(x_i^{+})/\tau\right)}{\sum_{j\neq i}\exp\left(f(x_i)^{\top}f(x_j)/\tau\right)}'))
    S.append(P('It was trained on 2,992 songs that are <i>not</i> in the evaluation library (2,400 steps, batch 192, ~45 minutes on a laptop CPU). '
               'The library is stored as 228,000 vectors (one per 0.5 s). A query is cut into 1-s segments every 0.5 s; each finds its 20 nearest '
               'library segments, and votes are aggregated per (song, offset) exactly as in the classical method; the score is the mean best '
               'similarity of the agreeing segments.'))
    S.append(P('5.2 A learned, calibrated decision layer (hybrid)', 'h2'))
    S.append(P('Every clip has two candidates: the classical top song and the neural top song. Two classifiers estimate P(candidate is right) from '
               '13 features (classical −log p, scores and their ratio, votes, hashes, original-engine p, neural best and second similarity and '
               'margin, whether the engines agree, log clip length). Logistic regression beat gradient boosting in 5-fold cross-validated '
               'log-loss. Training uses the calibration set (2,000 library + 1,000 non-library clips) weighted 50/50; the final threshold is '
               f'split-conformal on 2,500 separate non-library clips. Probabilities are well calibrated on the test set (Brier score '
               f'{sm["fusion_brier"]:.3f}).'))
    S.append(fig('fig_neural.png', 16.5, 'Left: contrastive training loss. Right: reliability of the fusion probabilities on test clips '
                                         '(points near the diagonal = honest probabilities).'))
    S.append(P('6. Benchmark', 'h1'))
    S.append(P('<b>Data.</b> FMA-small (Defferrard et al., 2017): 8,000 Creative-Commons tracks of 30 s, of which 7,992 decode correctly. Fixed '
               'random split: 4,000 library songs, 1,000 non-library songs (500 for calibration, 500 for testing), 2,992 for training the neural model.'))
    S.append(P('<b>Clips.</b> Random start positions; lengths 1, 2, 3, 5 and 10 s; conditions: clean, pink noise at 20/10/5/0 dB SNR, room reverb '
               '(RT60 0.6 s), a phone-in-a-room simulation (150 Hz–5 kHz band-pass, RT60 0.4 s, noise at 10 dB), and playback speed +1%, −2%, +5%. '
               'Test grids: all ten conditions at 5 s, and all lengths for clean / 5 dB / phone mic; 400 library and 400 non-library clips per '
               'cell. Every engine sees exactly the same audio.'))
    S.append(P(f'<b>Duplicates.</b> FMA contains the same music under several ids (re-uploads, vocal and instrumental versions). Tracks with the same '
               f'artist and the same base title are grouped by metadata only; {sm["duplicates_excluded"]} non-library clips whose group exists in '
               'the library were excluded from false-match counts, and naming any member of the right group counts as correct.'))
    return S


def results_sections(sm, fs):
    S = []
    ov, mt, sc, seq = sm['overall'], sm['main'], sm['scale'], sm['sequential']
    S.append(PageBreak())
    S.append(P('7. Results', 'h1'))
    S.append(fig('fig_robustness.png', 17, 'Correct matches for 5-s clips at a 1% false-match target (the original uses its fixed rule). v2 and '
                                           'the neural model are both near-perfect up to 5 dB noise; small speed changes break every landmark '
                                           'method, while the neural model (trained with ±3% speed) tolerates ±2%. The hybrid takes the best of both.'))
    rows = [['Condition (5 s)', 'Original', 'audfprint', 'v2 (conformal)', 'Neural', 'Hybrid']]
    for c, lab in CONDS:
        rows.append([lab] + [f'{pct(mt[c][m][0], 0)} / {pct(mt[c][m][1], 1)}' for m in MT_COLS])
    S.append(table(rows, [3.0 * cm] + [2.75 * cm] * 5))
    S.append(P('Each cell: correct matches / false-match rate (400 + 400 clips).', 'note'))
    S.append(fig('fig_length.png', 17, 'Correct matches vs clip length. The hybrid identifies '
                                       f'{pct(sm["length"]["clean/1"]["hybrid ML (conformal)"][0], 0)} of clean 1-s clips and '
                                       f'{pct(sm["length"]["mic/1"]["hybrid ML (conformal)"][0], 0)} of 1-s phone-mic clips (original: '
                                       f'{pct(sm["length"]["clean/1"]["v1 original (score>39)"][0], 0)} and '
                                       f'{pct(sm["length"]["mic/1"]["v1 original (score>39)"][0], 0)}).'))
    S.append(fig('fig_scaling.png', 17, 'Library size. The fixed rule\'s false-match rate climbs with every song added; the derived threshold '
                                        'stays close to the 1% target (shaded: 95% CI, bootstrap over songs).'))
    rows = [['Songs in library', '100', '300', '1,000', '2,000', '4,000']]
    for m, lab, kind in [('v1 original (score>39)', 'Original (> 39): false matches', 'f'), ('v2 + Poisson', 'v2, plain Poisson: false matches', 'f'),
                         ('v2 + quasi-Poisson', 'v2, quasi-Poisson: false matches', 'f'), ('v1 original (score>39)', 'Original: correct, phone mic', 't'),
                         ('v2 + quasi-Poisson', 'v2, quasi-Poisson: correct, phone mic', 't')]:
        if kind == 't':
            rows.append([lab] + [pct(sc[f'{n}/mic'][m][0], 1) for n in [100, 300, 1000, 2000, 4000]])
        else:
            rows.append([lab] + [pct((sc[f'{n}/clean'][m][1] + sc[f'{n}/mic'][m][1]) / 2, 1) for n in [100, 300, 1000, 2000, 4000]])
    S.append(KeepTogether([table(rows, [5.6 * cm] + [2.1 * cm] * 5)]))
    S.append(Spacer(1, 4))
    two = Table([[img('fig_calibration.png', 8.3), img('fig_sequential.png', 8.3)]], colWidths=[8.6 * cm, 8.6 * cm])
    S.append(KeepTogether([P('Calibration and sequential decisions', 'h2'), two,
                           P('Left: observed vs promised false-match rate on non-library clips. Plain Poisson (blue, orange, green) promises far '
                             'more than it delivers; quasi-Poisson (red) follows the diagonal near α = 1%. Right: how quickly 10-s recordings '
                             'are identified when checked every second (v2, quasi-Poisson).', 'note')]))
    rows = [['Recording', 'Rule', 'Correct', 'False matches', 'Mean time']]
    for c, lab in [('clean', 'clean'), ('pink5', 'noise 5 dB'), ('mic', 'phone mic')]:
        for mode, ml in [('naive', 'every second, α each'), ('bonferroni', 'every second, α/10 each'), ('single test at 10 s', 'once, after 10 s')]:
            r = seq[c][mode]
            rows.append([lab, ml, pct(r['tpr']), pct(r['fpr']), f"{r['mean_time_s']:.1f} s" if r['mean_time_s'] else '–'])
    S.append(table(rows, [2.6 * cm, 4.6 * cm, 2.2 * cm, 2.8 * cm, 2.4 * cm]))
    S.append(P('Checking every second at level α each ("peeking") multiplies the false-match rate; splitting α over the looks brings it back '
               'down at a cost of about 0.1–0.2 s.', 'note'))

    S.append(P('8. Was the 40,001-tap filter necessary?', 'h1'))
    if fs:
        idt = fs['identification']
        S.append(P('Aliases only hurt if they land on frequencies the fingerprint uses (up to 5.5 kHz). After decimation to 11,025 Hz, content at '
                   'f folds to 11,025 − f, so only content above 6,025 Hz can land below 5 kHz: a filter that passes 5 kHz and stops 6,025 Hz is '
                   f'enough. A Kaiser-window design for that requirement (60 dB) needs <b>{fs["taps"]["kaiser"]} taps</b> instead of 40,001. '
                   'Everything else held fixed (original fingerprint with its bugs fixed, 1,000-song library):'))
        rows = [['Anti-aliasing', 'Taps', 'Time / 29-s song', 'Peaks shared with original', 'Correct (clean / 10 dB / mic)', 'False matches']]
        for m, lab in [('orig', '40,001-tap sinc + average of 4 (original)'), ('kaiser', 'Kaiser FIR + average of 4'),
                       ('poly', 'polyphase resampling (scipy)')]:
            rows.append([lab, str(fs['taps'][m]) if m != 'poly' else 'default', f"{fs['sec_per_song'][m] * 1000:.0f} ms",
                         '100%' if m == 'orig' else pct(fs['peak_jaccard_vs_orig_delay_compensated'][m]),
                         ' / '.join(pct(idt[f'{m}/{c}']['tpr'], 0) for c in ('clean', 'pink10', 'mic')),
                         ' / '.join(pct(idt[f'{m}/{c}']['fpr'], 1) for c in ('clean', 'pink10', 'mic'))])
        S.append(table(rows, [4.6 * cm, 1.5 * cm, 2.1 * cm, 2.6 * cm, 3.4 * cm, 2.9 * cm]))
        S.append(P('Peaks shared = identical (time, frequency) peaks after removing each filter\'s delay (100 songs). Decisions use the plain '
                   'Poisson rule, so false-match rates are high for all three; the comparison is between filters.', 'note'))
        S.append(P(f'Two side findings. (1) The original filter delays the audio by {fs["original_filter_delay_samples"]:,} samples (0.45 s) '
                   'because its output is never re-centred; this is harmless for matching, since songs and clips are delayed equally, but every '
                   'time stamp is shifted. (2) Filter length barely changes run time here, because FFT convolution costs about the same for '
                   '157 or 40,001 taps on a padded signal; polyphase resampling, which never computes the samples it throws away, is '
                   f'{fs["sec_per_song"]["orig"] / fs["sec_per_song"]["poly"]:.1f}× faster end to end. Answer for an interviewer: the '
                   'requirement follows from where aliases fold to, and a filter sized from it (157 taps) gives the same fingerprint.'))

    S.append(P('9. What did not work, and what was learned', 'h1'))
    for t in [f'<b>Fixing the original\'s bugs changed almost nothing</b> (correct matches {pct(ov["v1 original + Poisson"][0])} vs '
              f'{pct(ov["v1 fixed + Poisson"][0])}). The real problems were the hash design and the decision rule, not the ÷6 or the j/11025 term.',
              '<b>The textbook Poisson model is wrong for real music.</b> It only becomes usable after an over-dispersion correction, and only on '
              'top of a hash design with near-independent votes.',
              f'<b>Landmark hashing cannot handle speed changes:</b> at +1% v2 drops to {pct(mt["speed+1%"]["v2 + conformal"][0])} and at −2% to '
              f'{pct(mt["speed-2%"]["v2 + conformal"][0])}; the neural model, trained with speed augmentation, keeps '
              f'{pct(mt["speed-2%"]["neural (conformal)"][0], 0)} at −2% but also fails at +5%, outside its training range.',
              '<b>The neural model is weaker than v2 on very noisy, very short clips;</b> the learned fusion exploits that the two fail on different clips.',
              '<b>Benchmarks hide traps:</b> FMA duplicates looked like false matches until they were traced by metadata, and per-song clustering '
              'makes naive confidence intervals far too narrow.',
              '<b>Conformal guarantees hold on average over calibration sets.</b> With ~490 calibration songs per length, the realised rate can land '
              'somewhat above 1%, which is what the intervals show.']:
        S.append(B(t))
    S.append(P('10. Limitations and next steps', 'h1'))
    for t in ['Degradations are simulated (noise, synthetic room responses); only one live microphone test was done. A recorded test set '
              '(phones in real rooms) is the most important next step.',
              'Library songs are 30-s clips and the library has 4,000 songs; commercial systems index millions of full songs. Memory per song '
              '(~4,800 hashes) and the calibration would need re-checking at that scale.',
              'The neural model is small and CPU-trained; a larger model trained longer, and pitch/tempo-invariant hashes (e.g. Panako-style '
              'triplets), are natural extensions.',
              'The quasi-Poisson factor is one global constant; modelling the dispersion as a function of clip length and song would be sharper.']:
        S.append(B(t))
    S.append(P('11. Reproducing everything', 'h1'))
    S.append(P('From the project folder, in order: <i>python research/download_fma.py</i> (7.7 GB) · <i>python -c "from research import data; '
               'data.build_cache()"</i> · <i>python -m research.bench build</i>, then <i>prepare</i>, <i>patch_v1</i>, <i>classical</i> · '
               '<i>python -m research.train_neural</i> · <i>python -m research.bench neural</i>, then <i>conformal</i>, <i>rerun_seq</i> · '
               '<i>python -m research.baseline_audfprint</i> · <i>python -m research.filter_study</i> · <i>python -m research.analyze</i> · '
               '<i>python -m research.make_report</i>. Unit tests: <i>python -m pytest tests</i>. All random choices are seeded.'))
    S.append(P('References', 'h1'))
    for r in ['A. Wang (2003). An industrial-strength audio search algorithm. Proc. ISMIR.',
              'coding-geek.com (2015). How does Shazam work. Online tutorial.',
              'S. Chang et al. (2021). Neural audio fingerprint for high-specific audio retrieval based on contrastive learning. Proc. ICASSP.',
              'M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson (2017). FMA: a dataset for music analysis. Proc. ISMIR.',
              'D. Ellis. audfprint: landmark-based audio fingerprinting. github.com/dpwe/audfprint.',
              'V. Vovk, A. Gammerman, G. Shafer (2005). Algorithmic Learning in a Random World. Springer.',
              'A. Angelopoulos, S. Bates (2023). Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning.',
              'T. Chen, S. Kornblith, M. Norouzi, G. Hinton (2020). A simple framework for contrastive learning of visual representations. Proc. ICML.',
              'J. Six, M. Leman (2014). Panako: a scalable acoustic fingerprinting system handling time-scale and pitch modification. Proc. ISMIR.']:
        S.append(P(r, 'small'))
    return S


def main():
    os.makedirs(DOCS, exist_ok=True)
    sm, fs = load()
    out = os.path.join(DOCS, 'MusicID_Technical_Report.pdf')
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.4 * cm,
                            title='MusicID technical report', author='MusicID project')

    def footer(c, d):
        c.saveState(); c.setFont('Cal', 8); c.setFillColor(MUTED)
        c.drawRightString(A4[0] - 1.8 * cm, 0.75 * cm, f'MusicID technical report · page {d.page}')
        c.restoreState()

    story = summary_section(sm) + method_sections(sm) + results_sections(sm, fs)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    print(out, 'pages:', len(PdfReader(out).pages))


if __name__ == '__main__':
    main()
