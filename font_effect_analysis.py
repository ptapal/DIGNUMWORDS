import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, wilcoxon
from itertools import combinations
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from rebuild_seqlevel import (
    load_seqlevel_h5, build_df, ANG_H5_OUT, TAL_H5_OUT,
)
from epoch_analysis import (
    build_records as build_erp_records,
    ANG_CSV_DIR, TAL_CSV_DIR, 
)
from rebuild_seqlevel import parse_angelique_fname, parse_talia_fname, _get_talia_files

OUT_DIR = 'results/font_effects'
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE = {'S20', 'S30'}
HIGHLIGHT = {
    ('Angelique', 'Dig',     '20F/A'): True,
    ('Angelique', 'NumWoGE', '20F/S'): True,
}

# load data
def load_snr_df():
    rec  = load_seqlevel_h5(ANG_H5_OUT, 'Angelique') + \
           load_seqlevel_h5(TAL_H5_OUT, 'Talia')
    df   = build_df(rec)
    df   = df[~df['subject'].isin(EXCLUDE)].reset_index(drop=True)
    return df

def load_erp_df():
    print('[diff-ERP] Loading Angelique')
    rec_ang = build_erp_records(ANG_CSV_DIR, parse_angelique_fname,
                                get_files_fn=None, dataset_name='Angelique')
    print('[diff-ERP] Loading Talia')
    rec_tal = build_erp_records(TAL_CSV_DIR, parse_talia_fname,
                                get_files_fn=_get_talia_files,
                                dataset_name='Talia')
    records = rec_ang + rec_tal
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'diff_erp'}
                       for r in records])
    return df

# compute per-subject effect sizes
def compute_effects(df_snr, df_erp):
    rows = []
    key_cols = ['dataset', 'subject', 'modality', 'font_label']

    for keys, grp in df_snr.groupby(key_cols):
        ds, subj, mod, font = keys
        par  = grp[grp['condition'] == 'Par']['snr_disc_all'].values
        ctrl = grp[grp['condition'] == 'Control']['snr_disc_all'].values
        if len(par) == 0 or len(ctrl) == 0:
            continue
        rows.append(dict(dataset=ds, subject=subj, modality=mod, font=font,
                         delta_snr=par.mean() - ctrl.mean(),
                         mean_par_snr=par.mean(), mean_ctrl_snr=ctrl.mean()))

    df_eff = pd.DataFrame(rows)

    # join diff-ERP deltas
    erp_rows = []
    for keys, grp in df_erp.groupby(key_cols):
        ds, subj, mod, font = keys
        par  = grp[grp['condition'] == 'Par']['rms_all'].values
        ctrl = grp[grp['condition'] == 'Control']['rms_all'].values
        if len(par) == 0 or len(ctrl) == 0:
            continue
        erp_rows.append(dict(dataset=ds, subject=subj, modality=mod, font=font,
                             delta_erp=par.mean() - ctrl.mean(),
                             mean_par_erp=par.mean(), mean_ctrl_erp=ctrl.mean()))

    df_erp_eff = pd.DataFrame(erp_rows)
    df_all = pd.merge(df_eff, df_erp_eff,
                      on=['dataset', 'subject', 'modality', 'font'],
                      how='outer')
    return df_all

# font ranking table
def rank_fonts(df_eff):
    rows = []
    for (ds, mod, font), grp in df_eff.groupby(['dataset', 'modality', 'font']):
        for feat, col in [('SNR', 'delta_snr'), ('diff-ERP', 'delta_erp')]:
            vals = grp[col].dropna().values
            if len(vals) < 3:
                continue
            t, p = ttest_1samp(vals, 0.0)
            d    = vals.mean() / (vals.std(ddof=1) + 1e-12)
            best = HIGHLIGHT.get((ds, mod, font), False)
            rows.append(dict(
                dataset=ds, modality=mod, font=font, feature=feat,
                n=len(vals),
                mean_delta=round(vals.mean(), 5),
                sem=round(vals.std(ddof=1) / np.sqrt(len(vals)), 5),
                t=round(t, 3), p=round(p, 4),
                cohen_d=round(d, 3),
                pct_pos=round((vals > 0).mean() * 100, 1),
                best_condition=best,
            ))
    return pd.DataFrame(rows).sort_values(
        ['dataset', 'modality', 'feature', 'mean_delta'], ascending=[True, True, True, False])

# forest plot
def forest_plot(ranking, feat_col, feat_label, out_path):
    sub     = ranking[ranking['feature'] == feat_label].copy()
    combos  = sub[['dataset', 'modality']].drop_duplicates().values.tolist()
    n_panels = len(combos)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f'Font effect on Par vs Control — {feat_label}\n'
                 'delta = mean(Par) - mean(Control) per subject; error = ±1 SEM',
                 fontsize=12, fontweight='bold')

    for ax, (ds, mod) in zip(axes, combos):
        panel = sub[(sub['dataset'] == ds) & (sub['modality'] == mod)].copy()
        panel = panel.sort_values('mean_delta', ascending=True)

        y   = np.arange(len(panel))
        ax.barh(y, panel['mean_delta'], xerr=panel['sem'],
                color=['goldenrod' if b else '#4393C3'
                       for b in panel['best_condition']],
                alpha=0.8, capsize=4, edgecolor='grey')

        for yi, (_, row) in enumerate(panel.iterrows()):
            sig = ('***' if row['p'] < 0.001 else '**' if row['p'] < 0.01
                   else '*' if row['p'] < 0.05 else 'ns')
            star = ' ★' if row['best_condition'] else ''
            ax.text(panel['mean_delta'].abs().max() * 1.15 + panel['sem'].max(),
                    yi,
                    f'{sig}{star}  d={row["cohen_d"]:.2f}  n={row["n"]}',
                    va='center', fontsize=8)

        ax.axvline(0, color='black', lw=1, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels(panel['font'].tolist(), fontsize=10)
        ax.set_xlabel(f'delta {feat_label} (Par - Ctrl)')
        ax.set_title(f'{ds} — {mod}', fontsize=11)
        ax.invert_yaxis() # best at top

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

# delta heatmap: subject x font
def delta_heatmap(df_eff, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('Per-subject effect size (Par - Control) by font\n'
                 'Blue = Control > Par; Red = Par > Control',
                 fontsize=12, fontweight='bold')

    for ax, col, title in [(axes[0], 'delta_snr', 'SNR@3.75 Hz'),
                            (axes[1], 'delta_erp', 'diff-ERP RMS')]:
        pivot = df_eff.pivot_table(
            index='subject',
            columns=['dataset', 'modality', 'font'],
            values=col, aggfunc='mean')

        # flatten column
        pivot.columns = [f'{d[:3]}/{m[:3]}/{f}' for d, m, f in pivot.columns]

        # order subjects consistently
        pivot = pivot.sort_index()

        vmax = np.nanpercentile(np.abs(pivot.values), 95)
        im   = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='delta (Par - Ctrl)')

        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha='right',
                           fontsize=7)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axhline(-0.5, color='white', lw=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'delta_heatmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# pairwise font comparisons
def pairwise_font_comparison(df_eff, out_dir):
    """
    For each (dataset, modality): Wilcoxon signed-rank test comparing every
    pair of fonts on per-subject delta_erp and delta_snr.
    """
    rows = []
    for (ds, mod), grp in df_eff.groupby(['dataset', 'modality']):
        fonts = grp['font'].unique()
        for fa, fb in combinations(sorted(fonts), 2):
            for feat, col in [('SNR', 'delta_snr'), ('diff-ERP', 'delta_erp')]:
                da = grp[grp['font'] == fa].set_index('subject')[col]
                db = grp[grp['font'] == fb].set_index('subject')[col]
                com = da.index.intersection(db.index)
                if len(com) < 5:
                    continue
                a, b = da[com].values, db[com].values
                try:
                    _, p = wilcoxon(a, b)
                except Exception:
                    p = 1.0
                rows.append(dict(
                    dataset=ds, modality=mod, feature=feat,
                    font_a=fa, font_b=fb,
                    mean_a=round(a.mean(), 4), mean_b=round(b.mean(), 4),
                    diff=round(a.mean() - b.mean(), 4),
                    n=len(com), p_wilcoxon=round(p, 4),
                    sig=('***' if p < 0.001 else '**' if p < 0.01
                         else '*' if p < 0.05 else 'ns'),
                ))
    df_pw = pd.DataFrame(rows).sort_values(['dataset', 'modality', 'feature', 'p_wilcoxon'])
    df_pw.to_csv(os.path.join(out_dir, 'pairwise_fonts.csv'), index=False)
    print(f'\nPairwise font comparisons:')
    for _, r in df_pw.iterrows():
        print(f'  {r["dataset"]:10} {r["modality"]:8} {r["feature"]:8}  '
              f'{r["font_a"]:8} vs {r["font_b"]:8}  '
              f'delta={r["diff"]:+.4f}  p={r["p_wilcoxon"]:.4f}  {r["sig"]}')
    return df_pw

# print summary
def print_ranking(ranking):
    print('\n' + '=' * 70)
    print('FONT EFFECT RANKING  (delta = Par - Control per subject)')
    print('=' * 70)
    for feat in ['SNR', 'diff-ERP']:
        print(f'\n  {feat}:')
        sub = ranking[ranking['feature'] == feat]
        for (ds, mod), grp in sub.groupby(['dataset', 'modality']):
            print(f'\n    {ds} — {mod}:')
            for _, r in grp.iterrows():
                sig  = ('***' if r['p'] < 0.001 else '**' if r['p'] < 0.01
                        else '*' if r['p'] < 0.05 else 'ns')
                star = ' ★' if r['best_condition'] else ''
                print(f'      {r["font"]:10}{star}  delta={r["mean_delta"]:+.4f}±{r["sem"]:.4f}  '
                      f't={r["t"]:+.2f}  p={r["p"]:.4f}  {sig}  d={r["cohen_d"]:+.3f}  '
                      f'{r["pct_pos"]:.0f}% Par>Ctrl')

# main
def main():
    print('=' * 70)
    print('Font effect analysis — effect-size approach (no clustering)')
    print('S20+S30 excluded; Angelique Dig+NumWoGE; Talia Dig')
    print('=' * 70)

    print('\nLoading SNR features')
    df_snr = load_snr_df()

    print('\nLoading diff-ERP features')
    df_erp = load_erp_df()

    print('\nComputing per-subject deltas')
    df_eff = compute_effects(df_snr, df_erp)
    df_eff.to_csv(os.path.join(OUT_DIR, 'effect_sizes.csv'), index=False)
    print(f'  {len(df_eff)} subject x font entries')

    ranking = rank_fonts(df_eff)
    ranking.to_csv(os.path.join(OUT_DIR, 'font_ranking.csv'), index=False)
    print_ranking(ranking)

    print('\nPairwise font comparisons')
    pairwise_font_comparison(df_eff, OUT_DIR)

    print('\nFigures')
    forest_plot(ranking, 'delta_erp', 'diff-ERP',
                os.path.join(OUT_DIR, 'forest_plot_erp.png'))
    forest_plot(ranking, 'delta_snr', 'SNR',
                os.path.join(OUT_DIR, 'forest_plot_snr.png'))
    delta_heatmap(df_eff, OUT_DIR)

    print(f'\nAll outputs -> {OUT_DIR}/')

if __name__ == '__main__':
    main()
