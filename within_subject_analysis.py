import os, sys, re, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_1samp, wilcoxon
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
    RANDOM, COND_COLORS,
)
from epoch_analysis import (
    build_records as build_erp_records,
    ANG_CSV_DIR, TAL_CSV_DIR,
)
from rebuild_seqlevel import (
    parse_angelique_fname, parse_talia_fname, _get_talia_files,
)

OUT_DIR = 'results/within_subject'
os.makedirs(OUT_DIR, exist_ok=True)

# subjects tha did not pass the data screening
EXCLUDE = {'S20', 'S30'}
DS_LABEL = {'Angelique': 'D1', 'Talia': 'D2'}

# helpers
def _purity(y_true, y_pred):
    """Cluster purity: fraction of points in majority class per cluster."""
    total = 0
    for cl in np.unique(y_pred):
        m = y_pred == cl
        total += max(y_true[m].sum(), m.sum() - y_true[m].sum())
    return total / len(y_true)

def empirical_chance_purity(n, n_A=None, n_repeats=2000, seed=RANDOM):
    """Monte Carlo chance level for purity with k=2, balanced conditions."""
    rng = np.random.default_rng(seed)
    if n_A is None:
        n_A = n // 2
    labels   = np.array([1] * (n // 2) + [0] * (n - n // 2))
    clusters = np.array([0] * n_A + [1] * (n - n_A))
    total = 0.0
    for _ in range(n_repeats):
        shuf = rng.permutation(labels)
        total += _purity(shuf, clusters)
    return total / n_repeats

def block_perm_purity(y_true, lbls, blocks, n_perm=10000, seed=RANDOM):
    """Permutation null for purity with the BLOCK as the exchangeable unit."""
    y_true = np.asarray(y_true)
    blocks = np.asarray(blocks)
    obs = _purity(y_true, lbls)
    ub = np.unique(blocks)
    if len(ub) < 2:
        return float('nan'), float('nan')
    blab = np.empty(len(ub), dtype=int)
    for i, b in enumerate(ub):
        v = y_true[blocks == b]
        assert (v == v[0]).all(), f'block {b} spans both conditions'
        blab[i] = v[0]
    idx = np.searchsorted(ub, blocks)
    rng  = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _purity(rng.permutation(blab)[idx], lbls)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return float(null.mean()), float(p)

def _cluster_and_score(X, y_true, n_components_pca=None, blocks=None):
    """
    Standardise -> optional PCA -> UMAP(2D) -> GMM/KMeans.
    Returns dict of scores and the 2D embedding.
    """
    n = X.shape[0]
    if n < 6:
        return None, None

    Xsc = StandardScaler().fit_transform(X.astype(np.float64))

    # PCA only if dims > samples
    if n_components_pca and X.shape[1] > n_components_pca:
        n_pc = min(n_components_pca, n - 1)
        Xsc  = PCA(n_components=n_pc, random_state=RANDOM).fit_transform(Xsc)

    n_nb = max(3, min(8, n // 4))
    emb  = umap.UMAP(n_neighbors=n_nb, min_dist=0.2,
                     random_state=RANDOM).fit_transform(Xsc)

    scores = {}
    for name, Model in [
        ('gmm',    GaussianMixture(n_components=2, n_init=5, random_state=RANDOM)),
        ('kmeans', KMeans(n_clusters=2, n_init=10, random_state=RANDOM)),
    ]:
        lbls = Model.fit_predict(emb)
        scores[f'purity_{name}']  = round(_purity(y_true, lbls), 4)
        scores[f'ari_{name}']     = round(adjusted_rand_score(y_true, lbls), 4)

        if blocks is not None:
            nm, p = block_perm_purity(y_true, lbls, blocks)
            scores[f'blocknull_{name}'] = round(nm, 4)
            scores[f'blockp_{name}']    = round(p, 4)
            # how much of the clustering is just block identity?
            bl    = np.asarray(blocks)
            ub    = np.unique(bl)
            coh   = np.mean([len(set(lbls[bl == b])) == 1 for b in ub])
            # baseline if cluster membership were independent of block, given the observed cluster sizes
            pc    = np.bincount(lbls, minlength=2) / len(lbls)
            coh0  = np.mean([(pc ** np.sum(bl == b)).sum() for b in ub])
            scores[f'cohesion_{name}']  = round(float(coh), 4)
            scores[f'cohesion0_{name}'] = round(float(coh0), 4)

    # silhouette by CLUSTER (label-free)
    if len(np.unique(emb[:, 0])) > 1:
        scores['silhouette']       = round(silhouette_score(emb, lbls), 4)
        scores['silhouette_label'] = round(silhouette_score(emb, y_true), 4)
    else:
        scores['silhouette'] = scores['silhouette_label'] = 0.0

    return scores, emb

# data load
def load_snr_data():
    """Load SNR features from H5, filter excluded subjects."""
    rec_ang = load_seqlevel_h5(ANG_H5_OUT, 'Angelique')
    rec_tal = load_seqlevel_h5(TAL_H5_OUT, 'Talia')
    df      = build_df(rec_ang + rec_tal)
    df      = df[~df['subject'].isin(EXCLUDE)].reset_index(drop=True)
    print(f'[SNR] {len(df)} sequences from {df["subject"].nunique()} subjects')
    return df

def load_erp_data():
    """Load diff-ERP features from raw CSVs (no condition labels used)."""
    print('[diff-ERP] Loading Angelique')
    rec_ang = build_erp_records(ANG_CSV_DIR, parse_angelique_fname,
                                get_files_fn=None, dataset_name='Angelique')
    print('[diff-ERP] Loading Talia')
    rec_tal = build_erp_records(TAL_CSV_DIR, parse_talia_fname,
                                get_files_fn=_get_talia_files,
                                dataset_name='Talia')
    records = rec_ang + rec_tal
    df_meta = pd.DataFrame([{k: v for k, v in r.items() if k != 'diff_erp'}
                             for r in records])
    print(f'[diff-ERP] {len(records)} sequences from '
          f'{df_meta["subject"].nunique()} subjects')
    return records, df_meta

# per-subject analysis
def analyse_snr_per_subject(df_snr):
    """Run within-subject clustering on all-electrode SNR features."""
    rows   = []
    embeds = {}

    for (ds, subj), sub in df_snr.groupby(['dataset', 'subject']):
        flat   = np.stack(sub['flat'].values).astype(np.float64)
        y_true = (sub['condition'] == 'Par').astype(int).values
        mod    = sub['modality'].iloc[0] if sub['modality'].nunique() == 1 else 'Mixed'

        if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
            continue

        sc, emb = _cluster_and_score(flat, y_true, n_components_pca=15, blocks=sub['block_id'].values)
        if sc is None:
            continue

        key = (ds, subj)
        rows.append(dict(subject=subj, dataset=ds, modality=mod,
                         n_seq=len(sub), feature='SNR_all', **sc))
        embeds[key] = (emb, y_true, ds, mod, sub)

    return pd.DataFrame(rows), embeds

def analyse_erp_per_subject(records, df_meta):
    """Run within-subject clustering on diff-ERP features (fully unsupervised)."""
    rows   = []
    embeds = {}

    for (ds, subj), grp in df_meta.groupby(['dataset', 'subject']):
        idx     = grp.index.tolist()
        recs_s  = [records[i] for i in idx]
        y_true  = np.array([(1 if r['condition'] == 'Par' else 0) for r in recs_s])
        mods    = list({r['modality'] for r in recs_s})
        mod     = mods[0] if len(mods) == 1 else 'Mixed'

        if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
            continue

        # full diff-ERP; labels never used here
        occ_ravel = np.stack([r['diff_erp'].ravel() for r in recs_s])
        blocks  = np.array([r['block_id'] for r in recs_s])
        sc, emb = _cluster_and_score(occ_ravel, y_true, n_components_pca=10, blocks=blocks)
        if sc is None:
            continue

        key = (ds, subj)
        rows.append(dict(subject=subj, dataset=ds, modality=mod,
                         n_seq=len(recs_s), feature='diffERP_all', **sc))
        embeds[key] = (emb, y_true, ds, mod, grp)

    return pd.DataFrame(rows), embeds

# plots
def plot_purity_histogram(scores_snr, scores_erp, out_dir):
    chance_line = empirical_chance_purity(24)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Within-subject clustering purity distribution\n'
                 f'Dashed line = empirical chance (n=24: {chance_line:.3f})',
                 fontsize=12, fontweight='bold')

    for ax, df, title in [
        (axes[0], scores_snr, 'SNR features (freq domain)'),
        (axes[1], scores_erp, 'diff-ERP features (time domain)'),
    ]:
        for name, color in [('gmm', '#D6604D'), ('kmeans', '#4393C3')]:
            col = f'purity_{name}'
            ax.hist(df[col], bins=12, range=(0, 1), alpha=0.6,
                    color=color, label=name.upper(), edgecolor='white')
            ax.axvline(df[col].mean(), color=color, lw=2, ls='--',
                       label=f'{name.upper()} mean={df[col].mean():.3f}')

        ax.axvline(chance_line, color='black', lw=1.5, ls=':',
                   label=f'empirical chance ({chance_line:.3f})')
        ax.set_xlabel('Purity')
        ax.set_ylabel('# subjects')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    out = os.path.join(out_dir, 'purity_histogram.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_purity_by_group(scores_snr, scores_erp, out_dir):
    """Bar chart of mean GMM purity per (dataset, modality)."""
    chance_24 = empirical_chance_purity(24)
    combined = pd.concat([
        scores_snr.assign(feature_type='SNR'),
        scores_erp.assign(feature_type='diff-ERP'),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('Mean GMM purity by group; within-subject clustering\n'
                 'Error bars = ±1 SEM across subjects', fontsize=12)

    for ax, ft in zip(axes, ['SNR', 'diff-ERP']):
        sub = combined[combined['feature_type'] == ft]
        grp = sub.groupby(['dataset', 'modality'])
        labels, means, sems, ns = [], [], [], []
        for (ds, mod), g in grp:
            labels.append(f'{DS_LABEL.get(ds, ds)}\n{mod}')
            means.append(g['purity_gmm'].mean())
            sems.append(g['purity_gmm'].sem())
            ns.append(len(g))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=sems, capsize=4,
                      color=['#D6604D' if 'D1' in l else '#4393C3' for l in labels],
                      alpha=0.75, edgecolor='grey')
        for xi, (m, n) in enumerate(zip(means, ns)):
            ax.text(xi, m + sems[xi] + 0.01, f'n={n}', ha='center',
                    fontsize=8, va='bottom')
        ax.axhline(chance_24, color='black', lw=1, ls='--',
                   label=f'empirical chance ({chance_24:.3f})')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('GMM purity'); ax.set_ylim(0, 1)
        ax.set_title(ft)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, 'purity_by_group.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_umap_grid(embeds, scores_df, feature_name, out_dir):
    """One UMAP panel per subject, sorted by GMM purity (best->worst)."""
    sorted_rows = scores_df.sort_values('purity_gmm', ascending=False)
    keys   = [(r.dataset, r.subject) for _, r in sorted_rows.iterrows()]
    n      = len(keys)
    ncols  = 6
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = axes.ravel() if n > 1 else [axes]

    fig.suptitle(f'Within-subject UMAP — {feature_name}\n'
                 'Sorted by GMM purity (best top-left); Red=Par  Blue=Ctrl',
                 fontsize=11, fontweight='bold')

    for pi, key in enumerate(keys):
        ax = axes_flat[pi]
        if key not in embeds:
            ax.set_visible(False)
            continue

        ds, subj   = key
        emb, y_true, ds, mod, _ = embeds[key]
        row = scores_df[(scores_df['dataset'] == ds) &
                        (scores_df['subject'] == subj)].iloc[0]

        for cval, cond, color in [(1, 'Par', COND_COLORS['Par']),
                                   (0, 'Ctrl', COND_COLORS['Control'])]:
            m = y_true == cval
            ax.scatter(emb[m, 0], emb[m, 1], c=color, s=22, alpha=0.7,
                       label=cond, rasterized=True)

        ax.set_title(f'{subj}  {DS_LABEL.get(ds, ds)}/{mod[:3]}\n'
                     f'pur={row.purity_gmm:.2f}  sil={row.silhouette:.2f}',
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        p = row.purity_gmm
        frame_color = ('#2ca02c' if p >= 0.70 else
                       '#ff7f0e' if p >= 0.60 else '#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2)

    for pi in range(len(keys), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, f'umap_grid_{feature_name.replace(" ", "_")}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_best_subjects(embeds_snr, embeds_erp, scores_snr, scores_erp, out_dir,
                       n_best=6):
    """Side-by-side SNR vs diff-ERP UMAP for the N subjects with highest purity."""
    top_rows = (scores_snr.sort_values('purity_gmm', ascending=False)
                .head(n_best)[['dataset', 'subject']].values.tolist())

    fig, axes = plt.subplots(n_best, 2, figsize=(10, n_best * 3.5))
    fig.suptitle(f'Top {n_best} subjects by SNR purity — SNR vs diff-ERP\n'
                 'Green frame = purity >= 0.70; Orange = purity >= 0.60',
                 fontsize=11, fontweight='bold')

    for ri, (ds, subj) in enumerate(top_rows):
        key = (ds, subj)
        for ci, (embeds, scores, label) in enumerate([
            (embeds_snr, scores_snr, 'SNR occ'),
            (embeds_erp, scores_erp, 'diff-ERP occ'),
        ]):
            ax = axes[ri, ci]
            if key not in embeds:
                ax.set_visible(False)
                continue

            emb, y_true, ds_, mod, _ = embeds[key]
            row = scores[(scores['dataset'] == ds) & (scores['subject'] == subj)]
            if row.empty:
                ax.set_visible(False)
                continue
            row = row.iloc[0]

            for cval, cond, color in [(1, 'Par', COND_COLORS['Par']),
                                       (0, 'Ctrl', COND_COLORS['Control'])]:
                m = y_true == cval
                ax.scatter(emb[m, 0], emb[m, 1], c=color, s=35, alpha=0.8,
                           label=cond, rasterized=True)

            ax.set_title(f'{subj} [{DS_LABEL.get(ds_, ds_)}/{mod}]  {label}\n'
                         f'purity={row.purity_gmm:.3f}  sil={row.silhouette:.3f}  '
                         f'ARI={row.ari_gmm:.3f}',
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=7, markerscale=1.5)

            p = row.purity_gmm
            fc = ('#2ca02c' if p >= 0.70 else '#ff7f0e' if p >= 0.60 else '#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor(fc); spine.set_linewidth(2.5)

    plt.tight_layout()
    out = os.path.join(out_dir, 'best_subjects_umap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_scatter_purity_vs_purity(scores_snr, scores_erp, out_dir):
    """Scatter: SNR purity vs diff-ERP purity per subject."""
    merged = pd.merge(
        scores_snr[['subject', 'dataset', 'modality', 'purity_gmm', 'silhouette']],
        scores_erp[['subject', 'dataset', 'purity_gmm', 'silhouette']],
        on=['subject', 'dataset'], suffixes=('_snr', '_erp'))

    fig, ax = plt.subplots(figsize=(7, 6))
    for ds, color in [('Angelique', '#D6604D'), ('Talia', '#4393C3')]:
        m = merged['dataset'] == ds
        ax.scatter(merged.loc[m, 'purity_gmm_snr'],
                   merged.loc[m, 'purity_gmm_erp'],
                   c=color, alpha=0.7, s=45, label=DS_LABEL.get(ds, ds), zorder=4)

    chance_24 = empirical_chance_purity(24)
    ax.axhline(chance_24, color='grey', lw=0.8, ls='--')
    ax.axvline(chance_24, color='grey', lw=0.8, ls='--')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='identity')

    corr = merged['purity_gmm_snr'].corr(merged['purity_gmm_erp'])
    ax.text(0.04, 0.96, f'r = {corr:.3f}', transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(fc='white', ec='grey', alpha=0.7))

    ax.set_xlabel('GMM purity - SNR features (freq domain)')
    ax.set_ylabel('GMM purity — diff-ERP features (time domain)')
    ax.set_title('Per-subject GMM purity', fontsize=10)
    ax.legend(); ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
    plt.tight_layout()
    out = os.path.join(out_dir, 'purity_agreement.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# modality & font breakdown
def _cluster_simple(X, y_true, n_components_pca=6):
    """
    PCA -> KMeans(k=2) only, no UMAP. Used when n is too small for UMAP.
    Returns purity or None.
    """
    n = X.shape[0]
    if n < 4 or y_true.sum() < 1 or (n - y_true.sum()) < 1:
        return None
    Xsc = StandardScaler().fit_transform(X.astype(np.float64))
    n_pc = min(n_components_pca, n - 1, X.shape[1])
    if n_pc >= 1:
        Xsc = PCA(n_components=n_pc, random_state=RANDOM).fit_transform(Xsc)
    lbls = KMeans(n_clusters=2, n_init=10, random_state=RANDOM).fit_predict(Xsc)
    return round(_purity(y_true, lbls), 4)

def breakdown_by_modality(df_snr, records_erp, df_erp_meta):
    """
    Per (dataset, subject, modality) — separates Dig from NumWoGE for Angelique.
    Uses the same UMAP+GMM pipeline as the main analysis.
    """
    rows_snr, rows_erp = [], []

    # SNR
    for (ds, subj, mod), sub in df_snr.groupby(['dataset', 'subject', 'modality']):
        flat   = np.stack(sub['flat'].values) # 612-D all electrodes
        y_true = (sub['condition'] == 'Par').astype(int).values
        if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
            continue
        sc, _ = _cluster_and_score(flat, y_true, n_components_pca=10, blocks=sub['block_id'].values)
        if sc:
            rows_snr.append(dict(subject=subj, dataset=ds, modality=mod,
                                 n_seq=len(sub), **sc))

    # diff-ERP
    for (ds, subj, mod), grp in df_erp_meta.groupby(['dataset', 'subject', 'modality']):
        idx       = grp.index.tolist()
        recs_s    = [records_erp[i] for i in idx]
        y_true    = np.array([1 if r['condition'] == 'Par' else 0 for r in recs_s])
        if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
            continue
        occ_ravel = np.stack([r['diff_erp'].ravel() for r in recs_s])
        sc, _ = _cluster_and_score(occ_ravel, y_true, n_components_pca=8, blocks=np.array([r['block_id'] for r in recs_s]))
        if sc:
            rows_erp.append(dict(subject=subj, dataset=ds, modality=mod,
                                 n_seq=len(recs_s), **sc))

    return pd.DataFrame(rows_snr), pd.DataFrame(rows_erp)

def breakdown_by_font(df_snr, records_erp, df_erp_meta):
    """
    Per (dataset, subject, modality, font_label).
    Typically only 4-6 sequences per cell — uses PCA+KMeans (no UMAP).
    """
    rows_snr, rows_erp = [], []

    # SNR
    for (ds, subj, mod, font), sub in df_snr.groupby(
            ['dataset', 'subject', 'modality', 'font_label']):
        flat   = np.stack(sub['flat'].values) # 612-D all electrodes
        y_true = (sub['condition'] == 'Par').astype(int).values
        pur    = _cluster_simple(flat, y_true)
        if pur is not None:
            rows_snr.append(dict(dataset=ds, subject=subj, modality=mod,
                                 font=font, n_seq=len(sub), purity_kmeans=pur))

    # diff-ERP
    for (ds, subj, mod, font), grp in df_erp_meta.groupby(
            ['dataset', 'subject', 'modality', 'font_label']):
        idx       = grp.index.tolist()
        recs_s    = [records_erp[i] for i in idx]
        y_true    = np.array([1 if r['condition'] == 'Par' else 0 for r in recs_s])
        occ_ravel = np.stack([r['diff_erp'].ravel() for r in recs_s])
        pur       = _cluster_simple(occ_ravel, y_true)
        if pur is not None:
            rows_erp.append(dict(dataset=ds, subject=subj, modality=mod,
                                 font=font, n_seq=len(recs_s), purity_kmeans=pur))

    return pd.DataFrame(rows_snr), pd.DataFrame(rows_erp)

def plot_modality_breakdown(rows_snr, rows_erp, out_dir):
    """Bar chart: mean purity per (dataset x modality) for SNR and diff-ERP."""
    chance_12 = empirical_chance_purity(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f'Within-subject purity by modality\n'
                 f'Error bars = ±1 SEM;  dashed = empirical chance (n=12: {chance_12:.3f})',
                 fontsize=12, fontweight='bold')

    palette = {'Dig': '#4393C3', 'NumWoGE': '#D6604D', 'Mixed': '#888888'}

    for ax, df, title in [(axes[0], rows_snr, 'SNR (freq domain)'),
                          (axes[1], rows_erp, 'diff-ERP (time domain)')]:
        grps   = df.groupby(['dataset', 'modality'])
        labels = [f'{DS_LABEL.get(ds, ds)}\n{mod}' for (ds, mod), _ in grps]
        means  = [g['purity_gmm'].mean() for _, g in grps]
        sems   = [g['purity_gmm'].sem()  for _, g in grps]
        colors = [palette.get(mod, '#888') for (_, mod), _ in grps]
        ns     = [len(g) for _, g in grps]

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.8,
               edgecolor='grey')
        grp_list = list(grps)
        for xi, (m, s, n) in enumerate(zip(means, sems, ns)):
            diffs = grp_list[xi][1]['purity_gmm'].values - chance_12
            t, p  = ttest_1samp(diffs, 0.0)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(xi, m + s + 0.01, f'n={n}\n{sig}', ha='center',
                    fontsize=8, va='bottom')

        ax.axhline(chance_12, color='black', lw=1.2, ls='--',
                   label=f'empirical chance ({chance_12:.3f})')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('GMM purity'); ax.set_ylim(0.3, 1.0)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, 'breakdown_modality.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_font_heatmap(font_snr, font_erp, out_dir):
    """
    Heatmap: rows = (dataset x modality), cols = font condition.
    Color = (purity - empirical_chance) / (1 - empirical_chance),
    i.e. normalised excess above empirical chance per cell size.
    0 = at chance (white), 1 = perfect (dark green).
    """
    # per-row empirical chance: D1 cells have n=4 (n_A=1), D2 cells have n=6
    chance_map = {'Angelique': empirical_chance_purity(4, n_A=1),
                  'Talia':     empirical_chance_purity(6)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Within-subject KMeans purity by font condition\n'
        'Colour = excess above empirical chance, normalised to [0, 1]  '
        '(D1 chance~0.75, D2 chance~0.70)',
        fontsize=11, fontweight='bold')

    for ax, df, title in [(axes[0], font_snr, 'SNR (freq domain)'),
                          (axes[1], font_erp, 'diff-ERP (time domain)')]:
        pivot = (df.groupby(['dataset', 'modality', 'font'])['purity_kmeans']
                   .mean()
                   .reset_index()
                   .pivot(index=['dataset', 'modality'], columns='font',
                          values='purity_kmeans'))
        ns = (df.groupby(['dataset', 'modality', 'font'])['purity_kmeans']
                .count()
                .reset_index()
                .pivot(index=['dataset', 'modality'], columns='font',
                       values='purity_kmeans'))

        # normalise each row by its empirical chance
        norm_vals = np.full_like(pivot.values, np.nan, dtype=float)
        for ri, (ds, mod) in enumerate(pivot.index):
            ch = chance_map.get(ds, empirical_chance_purity(6))
            row = pivot.values[ri].astype(float)
            norm_vals[ri] = np.where(np.isnan(row), np.nan,
                                     (row - ch) / (1.0 - ch))

        im = ax.imshow(norm_vals, vmin=0.0, vmax=1.0, cmap='YlGn', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('normalised excess above chance')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0 (chance)', '0.25', '0.50', '0.75', '1.0 (perfect)'])

        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha='right', fontsize=9)
        ylabels = [f'{DS_LABEL.get(ds, ds)} {mod[:6]}' for ds, mod in pivot.index]
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(ylabels, fontsize=9)

        for ri in range(pivot.shape[0]):
            for ci in range(pivot.shape[1]):
                v    = pivot.values[ri, ci]
                nv   = norm_vals[ri, ci]
                n    = ns.values[ri, ci] if not np.isnan(ns.values[ri, ci]) else 0
                if not np.isnan(v):
                    txt_color = 'white' if (not np.isnan(nv) and nv > 0.6) else 'black'
                    ax.text(ci, ri, f'{v:.2f}\n(n={int(n)})',
                            ha='center', va='center', fontsize=7, color=txt_color)

        # separator between D1 and D2 rows
        n_d1 = sum(1 for ds, _ in pivot.index if ds == 'Angelique')
        if 0 < n_d1 < pivot.shape[0]:
            ax.axhline(n_d1 - 0.5, color='white', lw=1.5)
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, 'breakdown_font_heatmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def print_breakdown_summary(mod_snr, mod_erp, font_snr, font_erp):
    # Empirical chance for the modality split (12 seqs per modality) and font cells (4 or 6 seqs)
    chance_12 = empirical_chance_purity(12)
    chance_6  = empirical_chance_purity(6)
    chance_4  = empirical_chance_purity(4, n_A=1)
    print(f'\nEmpirical chance: n=12→{chance_12:.3f}  n=6→{chance_6:.3f}  n=4(n_A=1)→{chance_4:.3f}')

    print('\n' + '=' * 70)
    print('MODALITY BREAKDOWN  [vs empirical chance for n=12]')
    print('=' * 70)
    for label, df in [('SNR', mod_snr), ('diff-ERP', mod_erp)]:
        print(f'\n  {label}:')
        for (ds, mod), g in df.groupby(['dataset', 'modality']):
            diffs = g['purity_gmm'].values - chance_12
            t, p  = ttest_1samp(diffs, 0.0)
            sig   = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f'    {ds:10} {mod:8}  n={len(g):2}  '
                  f'mean={g["purity_gmm"].mean():.3f}  sd={g["purity_gmm"].std():.3f}  '
                  f't={t:+.2f}  p={p:.3f}  {sig}')

    print('\n' + '=' * 70)
    print('FONT BREAKDOWN — cross-font paired contrasts (Wilcoxon signed-rank)')
    print(f'  Empirical chance per cell: n=6→{chance_6:.3f}  n=4(n_A=1)→{chance_4:.3f}')
    print(f'  Absolute cell means shown for reference; significance from paired differences')
    print('=' * 70)
    for label, df in [('SNR', font_snr), ('diff-ERP', font_erp)]:
        print(f'\n  {label}:')

        grp = df.groupby(['dataset', 'modality', 'font'])['purity_kmeans']
        for (ds, mod, font), g in grp:
            print(f'    {DS_LABEL.get(ds,ds):3} {mod:8} {font:8}  '
                  f'n={int(g.count()):2}  mean={g.mean():.3f}')

        # Paired cross-font contrasts
        print(f'\n    Paired contrasts (Wilcoxon; Bonferroni α=0.05/3=0.017):')
        for (ds, mod), sub in df.groupby(['dataset', 'modality']):
            fonts = sorted(sub['font'].unique())
            # pivot: rows=subject, cols=font
            pivot = sub.pivot_table(index='subject', columns='font',
                                    values='purity_kmeans')
            ref_font = '1F' if '1F' in pivot.columns else fonts[0]
            comparisons = [(f, ref_font) for f in fonts if f != ref_font]
            # also add 20F/A vs 20F/S if both exist
            if '20F/A' in pivot.columns and '20F/S' in pivot.columns:
                comparisons.append(('20F/A', '20F/S'))
            for fa, fb in comparisons:
                if fa not in pivot.columns or fb not in pivot.columns:
                    continue
                both = pivot[[fa, fb]].dropna()
                if len(both) < 4:
                    continue
                diff = both[fa].values - both[fb].values
                if (diff == 0).all():
                    print(f'      {DS_LABEL.get(ds,ds)} {mod:8}: '
                          f'{fa} − {fb}  all ties, skipped')
                    continue
                stat, p = wilcoxon(diff, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.017 else 'ns'
                print(f'      {DS_LABEL.get(ds,ds)} {mod:8}: '
                      f'{fa} − {fb}  W={stat:.0f}  p={p:.3f}  {sig}  '
                      f'(Δ={diff.mean():+.3f})')

# summary stats
def print_summary(scores_snr, scores_erp):
    print('\n' + '=' * 70)
    print('WITHIN-SUBJECT CLUSTERING SUMMARY')
    print('=' * 70)

    # Empirical chance for typical n (most subjects have 24 sequences)
    # Reported alongside so we can see margin above empirical chance
    typical_n   = 24
    chance_24   = empirical_chance_purity(typical_n)
    chance_12   = empirical_chance_purity(12)   # per-modality split in D1
    print(f'\nEmpirical chance level  n=24: {chance_24:.4f}  n=12: {chance_12:.4f}')

    for label, df in [('SNR features (freq domain)', scores_snr),
                      ('diff-ERP (time domain — fully unsupervised)', scores_erp)]:
        print(f'\n{label}')
        for col in ['purity_gmm', 'purity_kmeans', 'silhouette', 'ari_gmm']:
            vals = df[col].values
            if 'purity' in col:
                # use per-row empirical chance keyed by n_seq
                row_chance = df['n_seq'].apply(empirical_chance_purity).values
                diffs = vals - row_chance
                t, p  = ttest_1samp(diffs, 0.0)
                print(f'  {col:20} mean={vals.mean():.3f}  sd={vals.std():.3f}  '
                      f'chance≈{row_chance.mean():.3f}  margin={diffs.mean():+.3f}  '
                      f't={t:+.2f}  p={p:.4f}')
            else:
                t, p = ttest_1samp(vals, 0.0)
                print(f'  {col:20} mean={vals.mean():.3f}  sd={vals.std():.3f}  '
                      f't={t:+.2f}  p={p:.4f}')

        above = (df['purity_gmm'] > 0.60).mean() * 100
        print(f'  % subjects purity_gmm > 0.60 : {above:.1f}%')
        above70 = (df['purity_gmm'] > 0.70).mean() * 100
        print(f'  % subjects purity_gmm > 0.70 : {above70:.1f}%')

        print(f'\n  Per (dataset x modality) [vs empirical chance]:')
        for (ds, mod), g in df.groupby(['dataset', 'modality']):
            row_ch = g['n_seq'].apply(empirical_chance_purity).values
            diffs  = g['purity_gmm'].values - row_ch
            t2, p2 = ttest_1samp(diffs, 0.0)
            print(f'    {ds:10} {mod:8}  n={len(g):2}  '
                  f'mean={g["purity_gmm"].mean():.3f}  chance≈{row_ch.mean():.3f}  '
                  f't={t2:+.2f}  p={p2:.4f}')

def print_block_summary(scores_snr, scores_erp):
    print('\n' + '=' * 70)
    print('BLOCK-LEVEL PERMUTATION (labels shuffled across source files)')
    print('=' * 70)
    for label, df in [('SNR', scores_snr), ('diff-ERP', scores_erp)]:
        for ds, g in df.groupby('dataset'):
            print(f'\n  {label:9} {DS_LABEL.get(ds, ds)}  n={len(g)}')
            print(f'    purity_gmm      mean = {g.purity_gmm.mean():.3f}')
            print(f'    seq-level chance     = '
                  f'{g.n_seq.apply(empirical_chance_purity).mean():.3f}')
            print(f'    BLOCK-level null     = {g.blocknull_gmm.mean():.3f}')
            print(f'    block cohesion      = {g.cohesion_gmm.mean():.3f} '
                  f'(independence baseline {g.cohesion0_gmm.mean():.3f})')
            print(f'    ARI(clusters, condition) = {g.ari_gmm.mean():+.3f}')
            p = g.blockp_gmm.values
            # Benjamini-Hochberg, no extra dependency
            o = np.argsort(p); q = np.empty_like(p)
            q[o] = np.minimum.accumulate(
                (p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
            print(f'    subjects p<.05 uncorrected: {(p < .05).sum()}/{len(p)}')
            print(f'    subjects q<.05 (BH-FDR):    {(q < .05).sum()}/{len(p)}')

def main():
    print('=' * 70)
    print('Within-subject unsupervised clustering')
    print('SNR features (freq domain)  +  diff-ERP features (time domain)')
    print('=' * 70)

    print('\nLoading SNR features')
    df_snr = load_snr_data()

    print('\nLoading diff-ERP features')
    records_erp, df_erp_meta = load_erp_data()

    print('\nAnalysing per subject: SNR')
    scores_snr, embeds_snr = analyse_snr_per_subject(df_snr)

    print('\nAnalysing per subject: diff-ERP (fully unsupervised)')
    scores_erp, embeds_erp = analyse_erp_per_subject(records_erp, df_erp_meta)

    print_block_summary(scores_snr, scores_erp)

    # save raw scores
    all_scores = pd.concat([scores_snr, scores_erp], ignore_index=True)
    all_scores.to_csv(os.path.join(OUT_DIR, 'scores.csv'), index=False)
    print(f'\nSaved scores -> {OUT_DIR}/scores.csv')

    print('\nModality & font breakdown')
    mod_snr, mod_erp   = breakdown_by_modality(df_snr, records_erp, df_erp_meta)
    font_snr, font_erp = breakdown_by_font(df_snr, records_erp, df_erp_meta)
    print_breakdown_summary(mod_snr, mod_erp, font_snr, font_erp)

    # save breakdown scores
    mod_snr.assign(feature='SNR_occ',      level='modality').to_csv(
        os.path.join(OUT_DIR, 'breakdown_modality_snr.csv'), index=False)
    mod_erp.assign(feature='diffERP_occ',  level='modality').to_csv(
        os.path.join(OUT_DIR, 'breakdown_modality_erp.csv'), index=False)
    font_snr.assign(feature='SNR_occ',     level='font').to_csv(
        os.path.join(OUT_DIR, 'breakdown_font_snr.csv'), index=False)
    font_erp.assign(feature='diffERP_occ', level='font').to_csv(
        os.path.join(OUT_DIR, 'breakdown_font_erp.csv'), index=False)

    print('\nFigures')
    plot_purity_histogram(scores_snr, scores_erp, OUT_DIR)
    plot_purity_by_group(scores_snr, scores_erp, OUT_DIR)
    plot_umap_grid(embeds_snr, scores_snr, 'SNR_occ', OUT_DIR)
    plot_umap_grid(embeds_erp, scores_erp, 'diffERP_occ', OUT_DIR)
    plot_best_subjects(embeds_snr, embeds_erp, scores_snr, scores_erp, OUT_DIR)
    plot_scatter_purity_vs_purity(scores_snr, scores_erp, OUT_DIR)
    plot_modality_breakdown(mod_snr, mod_erp, OUT_DIR)
    plot_font_heatmap(font_snr, font_erp, OUT_DIR)

    print(f'\nAll outputs -> {OUT_DIR}/')

if __name__ == '__main__':
    main()
