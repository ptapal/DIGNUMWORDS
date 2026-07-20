import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from rebuild_seqlevel import (
    load_seqlevel_h5, build_df,
    ANG_H5_OUT, TAL_H5_OUT,
    N_FEATS, N_ELEC, RANDOM,
)

OUT_DIR = 'results/harmonic_comparison'
os.makedirs(OUT_DIR, exist_ok=True)
EXCLUDE = {'S20', 'S30'}

DISC_FEAT_IDX = [0, 2, 4, 5, 7, 9, 10] # discrimination harmonics + ratio
STIM_FEAT_IDX = [1, 3, 6, 8]           # stimulation harmonics

ALL_ELEC = list(range(N_ELEC))  

def _feat_cols(elec_list, feat_list):
    return [e * N_FEATS + f for e in elec_list for f in feat_list]


CONDITIONS = {
    'disc_all': dict(
        label='Discrimination harmonics\n(3.75 + 11.25 + 18.75 Hz); all electrodes',
        cols=_feat_cols(ALL_ELEC, DISC_FEAT_IDX),
    ),
    'stim_all': dict(
        label='Stimulation harmonics\n(7.5 + 15 Hz); all electrodes',
        cols=_feat_cols(ALL_ELEC, STIM_FEAT_IDX),
    ),
}

# clustering helpers
def _purity(y_true, y_pred):
    total = 0
    for cl in np.unique(y_pred):
        m = y_pred == cl
        total += max(y_true[m].sum(), m.sum() - y_true[m].sum())
    return total / len(y_true)


def _cluster_and_score(X, y_true, n_components_pca=15):
    n = X.shape[0]
    if n < 6:
        return None

    Xsc = StandardScaler().fit_transform(X.astype(np.float64))
    if X.shape[1] > n_components_pca:
        n_pc = min(n_components_pca, n - 1)
        Xsc  = PCA(n_components=n_pc, random_state=RANDOM).fit_transform(Xsc)

    n_nb = max(3, min(8, n // 4))
    emb  = umap.UMAP(n_neighbors=n_nb, min_dist=0.2,
                     random_state=RANDOM).fit_transform(Xsc)

    best_purity = 0.0
    for Model in [
        GaussianMixture(n_components=2, n_init=5, random_state=RANDOM),
        KMeans(n_clusters=2, n_init=10, random_state=RANDOM),
    ]:
        lbls   = Model.fit_predict(emb)
        purity = _purity(y_true, lbls)
        if purity > best_purity:
            best_purity = purity

    sil = silhouette_score(emb, y_true) if len(np.unique(emb[:, 0])) > 1 else 0.0
    return {'purity': round(best_purity, 4), 'silhouette': round(sil, 4)}

# main analysis
def run():
    print('Loading H5 data')
    rec_ang = load_seqlevel_h5(ANG_H5_OUT, 'Angelique')
    rec_tal = load_seqlevel_h5(TAL_H5_OUT, 'Talia')
    df      = build_df(rec_ang + rec_tal)
    df      = df[~df['subject'].isin(EXCLUDE)].reset_index(drop=True)
    print(f'  {len(df)} sequences, {df.groupby(["dataset","subject"]).ngroups} subjects')

    all_rows = []
    for cond_key, cond_info in CONDITIONS.items():
        print(f'\n{cond_key} ({len(cond_info["cols"])} features)')
        cols = cond_info['cols']
        for (ds, subj), sub in df.groupby(['dataset', 'subject']):
            flat   = np.stack(sub['flat'].values)[:, cols]
            y_true = (sub['condition'] == 'Par').astype(int).values
            if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
                continue
            sc = _cluster_and_score(flat, y_true)
            if sc is None:
                continue
            all_rows.append(dict(
                condition=cond_key,
                subject=subj,
                dataset=ds,
                modality=sub['modality'].iloc[0] if sub['modality'].nunique() == 1 else 'Mixed',
                n_seq=len(sub),
                **sc,
            ))

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUT_DIR, 'harmonic_comparison_scores.csv'), index=False)

    # summary statistics
    print('\n' + '='*72)
    print('HARMONIC COMPARISON — WITHIN-SUBJECT PURITY')
    print('='*72)
    print(f'{"Condition":<12}  {"n":>4}  {"mean":>6}  {"sd":>6}  {"t":>6}  {"p":>8}  '
          f'  {"pct>0.60":>8}  {"pct>0.70":>8}')
    print('-'*72)

    summary = {}
    for cond_key in CONDITIONS:
        sub = results[results['condition'] == cond_key]['purity']
        t, p = ttest_1samp(sub, 0.5)
        row = dict(n=len(sub), mean=sub.mean(), sd=sub.std(),
                   t=t, p=p,
                   pct60=(sub > 0.60).mean(),
                   pct70=(sub > 0.70).mean())
        summary[cond_key] = row
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f'{cond_key:<12}  {row["n"]:>4}  {row["mean"]:>6.3f}  {row["sd"]:>6.3f}  '
              f'{row["t"]:>6.2f}  {p:>8.4f} {sig:<4}  '
              f'{row["pct60"]:>8.1%}  {row["pct70"]:>8.1%}')

    # dataset breakdown
    print('\nBy dataset')
    print(f'{"Condition":<12}  {"Dataset":<12}  {"n":>4}  {"mean":>6}  {"p":>8}')
    print('-'*56)
    for cond_key in CONDITIONS:
        sub = results[results['condition'] == cond_key]
        for ds, grp in sub.groupby('dataset'):
            t, p = ttest_1samp(grp['purity'], 0.5)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f'{cond_key:<12}  {ds:<12}  {len(grp):>4}  {grp["purity"].mean():>6.3f}  '
                  f'{p:>8.4f} {sig}')
        print()

    # paired comparison: disc_all vs stim_all
    print('\nPaired t-test: disc_all vs stim_all')
    disc = results[results['condition'] == 'disc_all'].set_index(['dataset', 'subject'])['purity']
    stim = results[results['condition'] == 'stim_all'].set_index(['dataset', 'subject'])['purity']
    common = disc.index.intersection(stim.index)
    d_vals = disc.loc[common].values
    s_vals = stim.loc[common].values
    t, p = ttest_rel(d_vals, s_vals)
    delta = d_vals.mean() - s_vals.mean()
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f'  disc_all vs stim_all: delta mean = {delta:+.3f},  t = {t:+.2f},  p = {p:.4f} {sig}')

    _plot_comparison(results, summary)
    print(f'\nOutputs saved to {OUT_DIR}/')

def _plot_comparison(results, summary):
    cond_keys   = ['disc_all', 'stim_all']
    labels_plot = ['Disc\nall', 'Stim\nall']
    colors      = ['#D6604D', '#4393C3']

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle('Harmonic comparison — within-subject purity\n'
                 'Discrimination (3.75 + 11.25 + 18.75 Hz) vs Stimulation (7.5 + 15 Hz)',
                 fontsize=12)

    positions = np.arange(len(cond_keys))

    # Left: violin + strip
    ax = axes[0]
    for i, (ck, lbl, col) in enumerate(zip(cond_keys, labels_plot, colors)):
        vals = results[results['condition'] == ck]['purity'].values
        parts = ax.violinplot(vals, positions=[i], widths=0.6, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
        parts['cmedians'].set_color(col)
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(42).uniform(-0.12, 0.12, len(vals)),
                   vals, color=col, alpha=0.35, s=18, zorder=3)
        m = summary[ck]['mean']
        ax.scatter([i], [m], color=col, s=60, zorder=5, marker='D')
        ax.annotate(f'{m:.3f}', (i, m), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9, color=col, fontweight='bold')

    ax.axhline(0.5, color='grey', ls='--', lw=1, label='Chance (0.5)')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_ylabel('Purity (best of GMM / KMeans)')
    ax.set_ylim(0.35, 1.02)
    ax.set_title('Purity distribution per condition')
    ax.legend(fontsize=9)

    # Right: bar chart with SEM and significance stars
    ax2 = axes[1]
    means = [summary[ck]['mean'] for ck in cond_keys]
    sems  = [summary[ck]['sd'] / np.sqrt(summary[ck]['n']) for ck in cond_keys]
    for i, (pos, m, col) in enumerate(zip(positions, means, colors)):
        ax2.bar(pos, m, width=0.55, color=col, alpha=0.85, zorder=3)
    ax2.errorbar(positions, means, yerr=sems, fmt='none', color='black',
                 capsize=4, lw=1.5, zorder=4)
    ax2.axhline(0.5, color='grey', ls='--', lw=1)
    for i, ck in enumerate(cond_keys):
        p = summary[ck]['p']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax2.text(i, means[i] + sems[i] + 0.012, sig,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_plot, fontsize=10)
    ax2.set_ylim(0.45, 0.72)
    ax2.set_ylabel('Mean purity ± SEM')
    ax2.set_title('Mean purity (± SEM)  —  * vs chance 0.5')
    ax2.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#D6604D', label='Discrimination harmonics (3.75 + 11.25 + 18.75 Hz)'),
        Patch(facecolor='#4393C3', label='Stimulation harmonics (7.5 + 15 Hz)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'harmonic_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved: {out}')

if __name__ == '__main__':
    run()
