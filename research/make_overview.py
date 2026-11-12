"""Two-page overview of the whole project with all the maths (docs/MusicID_Project_Overview.pdf).

python -m research.make_overview
"""
import os, json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from research import make_report as R

ST = {
    'title': ParagraphStyle('t', fontName='Cal-B', fontSize=17, leading=20, textColor=R.INK),
    'sub': ParagraphStyle('s', fontName='Cal', fontSize=9, leading=11.5, textColor=R.MUTED),
    'h': ParagraphStyle('h', fontName='Cal-B', fontSize=10.8, leading=13, textColor=R.ACCENT, spaceBefore=5, spaceAfter=1),
    'b': ParagraphStyle('b', fontName='Cal', fontSize=8.9, leading=11.1, textColor=R.INK, spaceAfter=1.5),
    'n': ParagraphStyle('n', fontName='Cal-I', fontSize=7.9, leading=9.6, textColor=R.MUTED, spaceAfter=1),
    'c': ParagraphStyle('c', fontName='Cal', fontSize=7.9, leading=9.5, textColor=R.INK, alignment=TA_CENTER),
    'sm': ParagraphStyle('sm', fontName='Cal', fontSize=7.9, leading=9.5, textColor=R.INK),
}


def P(t, s='b'):
    return Paragraph(t, ST[s])


def eq(t):
    return R.eq(t, size=9.6)


def build(sm):
    ov, ci = sm['overall'], sm['overall_fpr_ci']
    S = [P('MusicID: how it works, with all the maths', 'title'),
         P('A song-recognition system built from first principles: a landmark fingerprint (after Wang, 2003), a statistically derived '
           'match decision, a neural fingerprint trained with contrastive learning, and a calibrated learned fusion, all benchmarked on '
           '4,000 Creative-Commons songs (FMA). Full details and every experiment: MusicID_Technical_Report.pdf.', 'sub'), Spacer(1, 4)]
    steps = ['Record<br/>mic, 44.1 kHz', 'Spectrogram<br/>resample + STFT', 'Peaks<br/>constellation map', 'Hashes<br/>target zones',
             'Votes<br/>time offsets', 'Decision<br/>p-value / conformal']
    pipe = Table([[P(f'<b>{i + 1}.</b> {s}', 'c') for i, s in enumerate(steps)]], colWidths=[2.95 * cm] * 6, hAlign='CENTER')
    pipe.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), R.SHADE), ('BOX', (0, 0), (-1, -1), 0.5, R.RULE),
                              ('LINEAFTER', (0, 0), (-2, -1), 0.5, R.RULE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                              ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    S.append(pipe)

    S.append(P('1. From sound to a spectrogram', 'h'))
    S.append(P('Audio at any sample rate is averaged to mono and resampled to F<sub>s</sub> = 11,025 Hz with a polyphase anti-aliasing filter '
               '(Nyquist 5,512.5 Hz). It is cut into frames of N = 1,024 samples every H = 512 samples (46.4 ms), each multiplied by a Hann '
               'window to reduce spectral leakage, and Fourier-transformed (FFT, O(N log N)). The log magnitude is kept:'))
    S.append(eq(r'X_t[k]=\sum_{n=0}^{N-1}x[tH+n]\,w[n]\,e^{-2\pi i kn/N},\quad w[n]=\frac{1}{2}\left(1-\cos\frac{2\pi n}{N}\right),\quad '
                r'S_t[k]=\log|X_t[k]|,\quad \Delta f=\frac{F_s}{N}=10.77\ \mathrm{Hz}'))
    S.append(P('2. Peaks and hashes', 'h'))
    S.append(P('A point (t, k) is a peak if it is the largest value in a 7-frame × 21-bin neighbourhood and above the median; at most 30 peaks '
               'per second are kept (215 Hz–5.4 kHz). These loud points survive noise and volume changes; together they form the '
               '<b>constellation map</b>. Each peak (anchor a) is paired with up to 6 later peaks p in its <b>target zone</b> '
               '(1 ≤ t<sub>p</sub> − t<sub>a</sub> ≤ 40 frames, |k<sub>p</sub> − k<sub>a</sub>| ≤ 100 bins), giving a 24-bit hash stored with the '
               'song id and anchor time:'))
    S.append(eq(r'h=k_a\cdot 2^{15}+k_p\cdot 2^{6}+(t_p-t_a),\qquad \mathrm{index}:\ h\ \mapsto\ \{(\mathrm{song},\,t_a)\}\ \ (\mathrm{sorted\ arrays,\ binary\ search})'))
    S.append(P('3. Votes: the time offset gives the match away', 'h'))
    S.append(P('Every query hash found in the index votes for (song, δ) with δ = t<sub>a</sub><super>db</super> − t<sub>a</sub><super>query</super>. '
               'For the right song all votes share one δ (where the clip starts in the song); random matches scatter. Each anchor votes once '
               'per (song, δ), and a song\'s score is its best window of three neighbouring offsets:'))
    S.append(eq(r'c_s=\max_{\delta}\left(N_{s,\delta-1}+N_{s,\delta}+N_{s,\delta+1}\right)'))
    S.append(P('4. Deciding: a threshold with a known error rate', 'h'))
    S.append(P('<b>Null model.</b> If the clip is not from song s, its M<sub>s</sub> votes fall at random over K<sub>s</sub> possible offsets, so a window '
               'count is about Poisson(wλ<sub>s</sub>), λ<sub>s</sub> = M<sub>s</sub>/K<sub>s</sub>, w = 3. The top score must beat every offset of every '
               'song, so its family-wise p-value is'))
    S.append(eq(r'p(c)\approx 1-\exp\left(-\sum_s K_s\Pr\left[\mathrm{Poisson}(w\lambda_s)\geq c\right]\right),\qquad \mathrm{match\ if}\ p(c)<\alpha=1\%'))
    S.append(P(f'<b>Over-dispersion.</b> Real votes come in clumps (a repeated musical pattern aligns many hashes at once), so the plain model promised '
               f'1% but gave {R.pct(sm["phi_effect"]["fpr_poisson"])} false matches. Quasi-Poisson divides counts by φ, estimated on held-out '
               f'non-library clips (φ = {sm["calibration"]["phi_hat"]}): c/φ ~ Poisson(wλ<sub>s</sub>/φ). <b>Conformal:</b> for any score, the '
               'threshold τ = s<sub>(k)</sub>, k = ⌈(1−α)(n+1)⌉, of n calibration non-library clips guarantees P(false match) ≤ α '
               '(one threshold per clip length). <b>Sequential:</b> the app checks every second; using α/T at each of T checks keeps the total ≤ α '
               '(Bonferroni): P(∃t: p<sub>t</sub> &lt; α/T) ≤ Σ α/T = α.'))
    S.append(P('5. Neural fingerprint (contrastive learning)', 'h'))
    S.append(P('A small CNN maps each 1-s log-mel segment to a unit vector f(x) of 128 numbers. It is trained on 2,992 songs not in the test '
               'library: pairs (x, x<super>+</super>) are a clean segment and a degraded copy (shift, noise, echo, EQ, ±3% speed), and the NT-Xent '
               'loss makes them the most similar pair in the batch:'))
    S.append(eq(r'\ell_i=-\log\frac{e^{f(x_i)^{\top}f(x_i^{+})/\tau}}{\sum_{j\neq i}e^{f(x_i)^{\top}f(x_j)/\tau}},\qquad \tau=0.05'))
    S.append(P('Search: nearest neighbours of each query segment among 228,000 library segments, then offset voting as in section 3.'))
    S.append(P('6. Learned fusion', 'h'))
    S.append(P('Two logistic regressions estimate the probability that the classical or the neural candidate is right from 13 evidence features '
               'x (−log p, scores, margins, agreement, clip length); the more probable candidate is reported if it passes the conformal threshold:'))
    S.append(eq(r'P(\mathrm{right}\mid x)=\sigma(\beta^{\top}x)=\frac{1}{1+e^{-\beta^{\top}x}},\qquad \hat\beta=\arg\min_\beta\ -\sum_i \omega_i\left[y_i\log\sigma_i+(1-y_i)\log(1-\sigma_i)\right]'))

    S.append(P('7. Results (4,000-song library, 17,600 test clips, 1% false-match target)', 'h'))
    rows = [['Method', 'Correct matches', 'False matches (95% CI)', 'Top-1']]
    for m in ['v1 original (score>39)', 'audfprint (conformal)', 'v2 + conformal', 'neural (conformal)', 'hybrid ML (conformal)']:
        r, c = ov[m], ci[m]
        rows.append([R.NICE[m], R.pct(r[0]), f'{R.pct(r[1])} ({R.pct(c[0])}–{R.pct(c[1])})', R.pct(r[2])])
    t = R.table(rows, [7.2 * cm, 2.6 * cm, 4.0 * cm, 1.8 * cm], highlight_rows=[len(rows) - 1])
    S.append(t)
    S.append(P('Ten recording conditions (clean, noise 20–0 dB, reverb, phone microphone, speed ±1–5%) × clip lengths 1–10 s; the open-source '
               'audfprint is the external baseline; CIs bootstrap over songs.', 'n'))
    S.append(R.fig('fig_robustness.png', 16.2, 'Correct matches for 5-s clips by recording condition.'))
    S.append(R.fig('fig_length.png', 16.2, 'Correct matches vs clip length: the hybrid needs about 2 s even through a simulated phone microphone.'))
    S.append(R.fig('fig_scaling.png', 16.2, 'Library size: the guessed rule makes more false matches with every song added; the derived '
                                           'threshold stays near the 1% target (shaded: 95% CI).'))
    S.append(P('8. What the original version taught', 'h'))
    S.append(P(f'An exact (bit-identical, unit-tested) replica of the first version showed that its guessed rule "score &gt; 39" names a song for '
               f'{R.pct(ov["v1 original (score>39)"][1])} of clips that are not in the library, rising with library size, and that its '
               'hand-written FFT is correct to 10<super>-9</super>. Its 40,001-tap anti-aliasing filter could be 157 taps: content above '
               '11,025 − 5,000 = 6,025 Hz is all that can alias onto the bins used, and the shorter filter gives 98.5% identical peaks.'))
    S.append(P('Limits: simulated degradations, 30-s songs, a 4,000-song library and a small CPU-trained network; next steps are real phone '
               'recordings, full-length songs at larger scale and pitch/tempo-invariant hashes.', 'n'))
    return S


def main():
    sm = json.load(open(os.path.join(R.RESULTS, 'summary.json')))
    out = os.path.join(R.DOCS, 'MusicID_Project_Overview.pdf')
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title='MusicID project overview', author='MusicID project')

    def footer(c, d):
        c.saveState(); c.setFont('Cal', 7.5); c.setFillColor(R.MUTED)
        c.drawRightString(A4[0] - 1.6 * cm, 0.65 * cm, f'MusicID overview · page {d.page}')
        c.restoreState()

    doc.build(build(sm), onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    print(out, 'pages:', len(PdfReader(out).pages))


if __name__ == '__main__':
    main()
