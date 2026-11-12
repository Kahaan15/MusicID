"""Simple headline chart for the README: how often each approach finds the right song.

python -m research.make_readme_chart   ->  docs/images/accuracy.png
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from research.bench import RESULTS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'
ACCENT, GREY = '#2a78d6', '#cfcec8'


def main():
    sm = json.load(open(os.path.join(RESULTS, 'summary.json')))
    ov = sm['overall']
    rows = [('Hybrid: classical + neural', ov['hybrid ML (conformal)'], True),
            ('Neural fingerprint', ov['neural (conformal)'], False),
            ('Classical fingerprint', ov['v2 + conformal'], False),
            ('audfprint (open source)', ov['audfprint (conformal)'], False)]

    plt.rcParams['font.family'] = 'Segoe UI'
    fig, ax = plt.subplots(figsize=(8, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.58
    ax.set_xlim(0, 1.18)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.04)
    fig.canvas.draw()
    # rounded data-end (~4 px): convert pixels to data units on each axis
    bb = ax.get_window_extent()
    px_x = 1.18 / bb.width
    px_y = (len(rows) + 0.35) / bb.height
    r = 4 * px_x
    for i, (label, vals, hl) in enumerate(rows):
        v = vals[0]
        color = ACCENT if hl else GREY
        ax.add_patch(plt.Rectangle((0, i - h / 2), max(0, v - r), h, color=color, lw=0))
        ax.add_patch(FancyBboxPatch((max(0, v - 2 * r), i - h / 2), 2 * r, h, color=color, lw=0,
                                    boxstyle=f'round,pad=0,rounding_size={r}', mutation_aspect=px_y / px_x))
        ax.text(v + 0.012, i, f'{v * 100:.0f}%', va='center', ha='left', fontsize=12.5,
                color=INK, fontweight='bold' if hl else 'normal')
        ax.text(-0.02, i, label, va='center', ha='right', fontsize=10.5, color=INK if hl else INK2,
                fontweight='bold' if hl else 'normal', transform=ax.transData, clip_on=False)
    ax.axvline(0, color='#d9d8d2', lw=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.03, 0.89, 'MusicID finds the right song 94% of the time', fontsize=15.5, fontweight='bold', color=INK)
    out = os.path.join(ROOT, 'docs', 'images', 'accuracy.png')
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
