import os, sys, re, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import umap
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from rebuild_seqlevel import (
    parse_angelique_fname, parse_talia_fname,
    _get_talia_files, N_ELEC, RANDOM, COND_COLORS,
)

# paths
ANG_CSV_DIR = 'data/angelique_v2/unlabaled_data_angelique_v2'
TAL_CSV_DIR = 'data/talia/unlabeled_data_talia'
OUT_DIR     = 'results/epoch_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# constants
SR              = 512
STIM_RATE       = 7.5                            
EPOCH_SAMPLES   = int(round(SR / STIM_RATE)) # 68 samples ~ 133 ms
EXCLUDE_SUBJ    = {'S20', 'S30'}

# highlighted best conditions
HIGHLIGHT = {
    ('Angelique', 'Dig',     '20F/A'): True,
    ('Angelique', 'NumWoGE', '20F/S'): True,
}

# CSV loader
def _load_raw_sequences(fpath, n_elec=N_ELEC):
    """
    Load raw EEG CSV -> dict {cond_idx: ndarray (N, n_elec)}.
    CSV columns: time, channel, condition, value
    """
    df = pd.read_csv(fpath, dtype={'time': np.int32, 'channel': np.int8,
                                   'condition': np.int8, 'value': np.float32})
    result = {}
    for cond_idx in sorted(df['condition'].unique()):
        sub = df[df['condition'] == cond_idx]
        pivoted = (sub.pivot(index='time', columns='channel', values='value')
                      .values[:, :n_elec])
        result[int(cond_idx)] = pivoted.astype(np.float32)
    del df; gc.collect()
    return result

# epoch computation
def compute_diff_erp(eeg, epoch_samples=EPOCH_SAMPLES):
    N, n_elec = eeg.shape
    n_epochs  = N // epoch_samples
    # truncate to complete epochs
    eeg_trunc = eeg[:n_epochs * epoch_samples]
    # reshape to (n_epochs, epoch_samples, n_elec)
    epochs    = eeg_trunc.reshape(n_epochs, epoch_samples, n_elec)

    even_idx  = np.arange(0, n_epochs, 2) # 0, 2, 4, ...  (Set A)
    odd_idx   = np.arange(1, n_epochs, 2) # 1, 3, 5, ...  (Set B)

    mean_even = epochs[even_idx].mean(axis=0) # (epoch_samples, n_elec)
    mean_odd  = epochs[odd_idx].mean(axis=0)

    diff_erp  = mean_odd - mean_even # discrimination signal
    return diff_erp, len(even_idx), len(odd_idx)

def diff_erp_rms(diff_erp):
    """Per-electrode RMS of diff_ERP — scalar discrimination strength."""
    return np.sqrt((diff_erp ** 2).mean(axis=0)) # (n_elec,)

# dataset builder
def build_records(csv_dir, parse_fn, get_files_fn, dataset_name,
                  exclude_subj=EXCLUDE_SUBJ):
    if get_files_fn:
        files = get_files_fn(csv_dir)
    else:
        files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))

    records    = []
    seq_counts = {}

    for fname in sorted(files):
        fpath = os.path.join(csv_dir, fname)
        try:
            subj, mod, ftype, sfont, cond = parse_fn(fname)
        except Exception as e:
            print(f'  SKIP {fname}: {e}')
            continue

        if None in (subj, mod, ftype, cond):
            continue

        subj_key = f'S{subj}'
        if subj_key in exclude_subj:
            continue

        try:
            seq_data = _load_raw_sequences(fpath)
        except Exception as e:
            print(f'  SKIP {fname}: {e}')
            continue

        for seq_i, eeg in seq_data.items():
            diff_erp, ne, no = compute_diff_erp(eeg)

            rms_all = diff_erp_rms(diff_erp).mean()

            font_label = ftype
            if sfont:
                font_label = f'{ftype}/{sfont}'

            k = (subj_key, mod, font_label, cond)
            seq_counts[k] = seq_counts.get(k, 0) + 1

            records.append(dict(
                dataset     = dataset_name,
                subject     = subj_key,
                modality    = mod,
                font_type   = ftype,
                specific_font = sfont,
                font_label  = font_label,
                condition   = cond,
                seq_idx     = seq_counts[k] - 1,
                diff_erp    = diff_erp.astype(np.float32),
                rms_all     = float(rms_all),
                n_even_epochs = ne,
                n_odd_epochs  = no,
            ))

    print(f'[{dataset_name}] {len(records)} sequences from '
          f'{len({r["subject"] for r in records})} subjects')
    return records

# plot 1: mean |diff_ERP| per condition
def plot_rms_comparison(df, out_dir):
    """Violin + per-subject scatter of diff-ERP RMS (all electrodes), Par vs Control."""
    combos = df[['dataset', 'modality']].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(len(combos), 2, figsize=(13, 5 * len(combos)),
                             squeeze=False)
    fig.suptitle(
        'diff-ERP RMS (all electrodes) — Par vs Control\n'
        'S20+S30 excluded; epoch at 7.5 Hz; diff = odd - even position',
        fontsize=12, fontweight='bold')

    stat_rows = []
    for ri, (ds, mod) in enumerate(combos):
        sub   = df[(df['dataset'] == ds) & (df['modality'] == mod)]
        ax_v  = axes[ri, 0]
        ax_p  = axes[ri, 1]
        title = f'{ds} — {mod}'

        for xi, cond in enumerate(['Par', 'Control']):
            vals  = sub[sub['condition'] == cond]['rms_all'].values
            parts = ax_v.violinplot(vals, [xi], widths=0.5, showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(COND_COLORS[cond]); pc.set_alpha(0.5)
            parts['cmeans'].set_color('black')
            rng = np.random.default_rng(RANDOM)
            ax_v.scatter(
                np.full(len(vals), xi) + rng.uniform(-0.1, 0.1, len(vals)),
                vals, c=COND_COLORS[cond], alpha=0.35, s=14, zorder=3)

        ax_v.set_xticks([0, 1])
        ax_v.set_xticklabels(['Parity', 'Control'])
        ax_v.set_ylabel('diff-ERP RMS (µV, all elec)')
        ax_v.set_title(title)

        par_s  = sub[sub['condition'] == 'Par'].groupby('subject')['rms_all'].mean()
        ctrl_s = sub[sub['condition'] == 'Control'].groupby('subject')['rms_all'].mean()
        com    = par_s.index.intersection(ctrl_s.index)
        if len(com) >= 3:
            pv, cv = par_s[com].values, ctrl_s[com].values
            ax_p.scatter(cv, pv, c='steelblue', alpha=0.7, s=40, zorder=4)
            lo = min(np.concatenate([pv, cv])) * 0.97
            hi = max(np.concatenate([pv, cv])) * 1.03
            ax_p.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
            t, p = ttest_rel(pv, cv)
            d    = (pv - cv).mean() / ((pv - cv).std(ddof=1) + 1e-12)
            pct  = (pv > cv).mean() * 100
            ax_p.text(0.04, 0.96,
                      f'n={len(com)} subj\nt={t:.2f} p={p:.3f}\nd={d:.2f}  {pct:.0f}% Par>Ctrl',
                      transform=ax_p.transAxes, va='top', fontsize=8,
                      bbox=dict(fc='white', alpha=0.7, ec='grey'))
            stat_rows.append(dict(
                dataset=ds, modality=mod, font='ALL', n_subjects=len(com),
                t=round(t, 4), p=round(p, 4), d=round(d, 4),
                pct=round(pct, 1)))
        ax_p.set_xlabel('Control RMS'); ax_p.set_ylabel('Parity RMS')
        ax_p.set_title(title + ' per-subject')

    plt.tight_layout()
    out = os.path.join(out_dir, 'rms_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return pd.DataFrame(stat_rows)


# plot 2: per-font RMS with highlighted best conditions
def plot_per_font_rms(df, out_dir):
    combos     = df[['dataset', 'modality']].drop_duplicates().values.tolist()
    font_types = sorted(df['font_label'].unique())
    fig, axes  = plt.subplots(len(combos), len(font_types),
                              figsize=(5 * len(font_types), 4.5 * len(combos)),
                              squeeze=False)
    fig.suptitle('diff-ERP RMS (all electrodes) by Font — Par vs Control\n'
                 '★ = best condition per FPVS analysis', fontsize=12)

    stat_rows = []
    for ri, (ds, mod) in enumerate(combos):
        sub = df[(df['dataset'] == ds) & (df['modality'] == mod)]
        for ci, font in enumerate(font_types):
            ax   = axes[ri, ci]
            fsub = sub[sub['font_label'] == font]
            best = HIGHLIGHT.get((ds, mod, font), False)

            if fsub.empty:
                ax.set_visible(False)
                continue

            for xi, cond in enumerate(['Par', 'Control']):
                vals = fsub[fsub['condition'] == cond]['rms_all'].values
                if not len(vals): continue
                parts = ax.violinplot(vals, [xi], widths=0.5, showmeans=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(COND_COLORS[cond]); pc.set_alpha(0.5)
                ax.scatter(
                    np.full(len(vals), xi) +
                    np.random.default_rng(RANDOM).uniform(-0.1, 0.1, len(vals)),
                    vals, c=COND_COLORS[cond], alpha=0.3, s=12)

            pv  = fsub[fsub['condition'] == 'Par'].groupby('subject')['rms_all'].mean()
            cv  = fsub[fsub['condition'] == 'Control'].groupby('subject')['rms_all'].mean()
            com = pv.index.intersection(cv.index)
            stat_txt = ''
            if len(com) >= 3:
                t, p = ttest_rel(pv[com].values, cv[com].values)
                d    = (pv[com].values - cv[com].values).mean() / \
                       ((pv[com].values - cv[com].values).std(ddof=1) + 1e-12)
                pct  = (pv[com].values > cv[com].values).mean() * 100
                stat_txt = f't={t:.2f} p={p:.3f} d={d:.2f}'
                stat_rows.append(dict(
                    dataset=ds, modality=mod, font=font,
                    n_subjects=len(com),
                    t=round(t, 4), p=round(p, 4), d=round(d, 4),
                    pct=round(pct, 1),
                    best_condition=best))

            star = ' ★' if best else ''
            ax.set_title(f'{ds} {mod}\n{font}{star}\n{stat_txt}', fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Par', 'Ctrl'], fontsize=8)
            if ci == 0:
                ax.set_ylabel('RMS all elec (µV)')

            if best:
                for spine in ax.spines.values():
                    spine.set_edgecolor('goldenrod')
                    spine.set_linewidth(2.5)

    plt.tight_layout()
    out = os.path.join(out_dir, 'per_font_rms.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return pd.DataFrame(stat_rows)

# plot 3: mean diff_ERP waveforms per condition
def plot_diff_erp_waveforms(records, out_dir):
    """
    Grand-average diff_ERP waveform for Par vs Control, split by dataset.
    """
    combos = sorted({(r['dataset'], r['modality']) for r in records})
    t_axis = np.arange(EPOCH_SAMPLES) / SR * 1000 # ms

    fig, axes = plt.subplots(len(combos), 1, figsize=(10, 4 * len(combos)),
                             squeeze=False)
    fig.suptitle('Grand-average diff_ERP (odd - even)\n'
                 'All-electrode average; shading = ±1 SEM across sequences',
                 fontsize=12, fontweight='bold')

    for ri, (ds, mod) in enumerate(combos):
        ax = axes[ri, 0]
        for cond in ['Par', 'Control']:
            erps = np.stack([
                r['diff_erp'].mean(axis=1)
                for r in records
                if r['dataset'] == ds and r['modality'] == mod
                   and r['condition'] == cond
            ]) # (n_seq, epoch_samples)
            if erps.shape[0] == 0:
                continue
            mn  = erps.mean(axis=0)
            sem = erps.std(axis=0) / np.sqrt(erps.shape[0])
            ax.plot(t_axis, mn, color=COND_COLORS[cond], lw=1.8, label=cond)
            ax.fill_between(t_axis, mn - sem, mn + sem,
                            color=COND_COLORS[cond], alpha=0.2)

        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('Time within epoch (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'{ds} — {mod}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'diff_erp_waveforms.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# plot 4: UMAP on diff_ERP features
def plot_umap_diff_erp(records, df, out_dir):
    """
    Three feature representations:
      A) Full diff_ERP ravel (68 x 68 = 4624)
      B) Per-electrode RMS (68,)
    For each: subject-centered -> StandardScaler -> PCA(min(n,50)) -> UMAP
    """
    n = len(records)
    # stack features
    full_ravel = np.stack([r['diff_erp'].ravel() for r in records]) # (n, 4624)
    rms_all    = np.stack([diff_erp_rms(r['diff_erp']) for r in records]) # (n, 68)

    feat_sets = [
        ('Full diff-ERP (4624)', full_ravel),
        ('Per-elec RMS (68)',    rms_all),
    ]

    colorings = [
        ('Condition', 'condition', COND_COLORS),
        ('Subject',   'subject',   None),
        ('Dataset',   'dataset',   None),
    ]

    fig, axes = plt.subplots(len(feat_sets), len(colorings),
                             figsize=(7 * len(colorings), 6 * len(feat_sets)))
    fig.suptitle('UMAP on diff_ERP (odd-even epochs); S20+S30 excluded\n'
                 'Par = red; Control = blue',
                 fontsize=13, fontweight='bold')

    subjects = [r['subject'] for r in records]

    for ri, (feat_label, feat) in enumerate(feat_sets):
        print(f'  UMAP: {feat_label}')

        # subject-centering
        fc = feat.copy().astype(np.float64)
        for subj in set(subjects):
            m = np.array([s == subj for s in subjects])
            fc[m] -= fc[m].mean(axis=0)

        Xsc  = StandardScaler().fit_transform(fc)
        n_pc = min(50, Xsc.shape[0] - 1, Xsc.shape[1])
        Xpc  = PCA(n_components=n_pc, random_state=RANDOM).fit_transform(Xsc)
        emb  = umap.UMAP(n_neighbors=12, min_dist=0.1,
                         random_state=RANDOM).fit_transform(Xpc)
        del fc, Xsc, Xpc; gc.collect()

        for ci, (col_title, col_key, cmap_dict) in enumerate(colorings):
            ax   = axes[ri, ci]
            cats = list(dict.fromkeys(df[col_key])) # preserve order
            if cmap_dict:
                cat_colors = cmap_dict
            else:
                pal = plt.cm.get_cmap('tab20', max(len(cats), 1))
                cat_colors = {c: pal(i) for i, c in enumerate(cats)}

            for cat in cats:
                mask = (df[col_key] == cat).values
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[cat_colors.get(cat, 'grey')],
                           alpha=0.5, s=14, label=str(cat), rasterized=True)

            ax.set_title(f'{col_title}\n({feat_label})', fontsize=9)
            ax.set_xlabel('UMAP 1', fontsize=7)
            ax.set_ylabel('UMAP 2', fontsize=7)
            ax.tick_params(labelsize=6)
            if len(cats) <= 15:
                ax.legend(markerscale=2, fontsize=6, framealpha=0.5, ncol=2)

    plt.tight_layout()
    out = os.path.join(out_dir, 'umap_diff_erp.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# plot 5: UMAP per-condition subset with clustering
def plot_umap_clustered(records, df, out_dir):
    """
    UMAP on full diff-ERP, then GMM(k=2) and KMeans(k=2) purity.
    """
    full_ravel = np.stack([r['diff_erp'].ravel() for r in records])
    subjects   = [r['subject'] for r in records]

    fc = full_ravel.copy().astype(np.float64)
    for subj in set(subjects):
        m = np.array([s == subj for s in subjects])
        fc[m] -= fc[m].mean(axis=0)

    Xsc  = StandardScaler().fit_transform(fc)
    n_pc = min(50, Xsc.shape[0] - 1, Xsc.shape[1])
    Xpc  = PCA(n_components=n_pc, random_state=RANDOM).fit_transform(Xsc)
    emb  = umap.UMAP(n_neighbors=12, min_dist=0.1,
                     random_state=RANDOM).fit_transform(Xpc)
    del fc, Xsc, Xpc; gc.collect()

    y_true   = (df['condition'] == 'Par').astype(int).values
    results  = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Full diff-ERP; GMM and KMeans purity\n'
                 'Par=red circle; Control=blue triangle',
                 fontsize=12, fontweight='bold')
    markers = ['o', '^']

    for col_i, (Model, mname) in enumerate([
        (GaussianMixture(n_components=2, n_init=5, random_state=RANDOM), 'GMM k=2'),
        (KMeans(n_clusters=2, n_init=10, random_state=RANDOM), 'KMeans k=2'),
    ]):
        ax   = axes[col_i]
        lbls = Model.fit_predict(emb)
        purity = sum(
            max((y_true[lbls == cl]).sum(),
                (lbls == cl).sum() - (y_true[lbls == cl]).sum())
            for cl in np.unique(lbls)
        ) / len(y_true)

        for xi, cond in enumerate(['Par', 'Control']):
            tm = (df['condition'] == cond).values
            for cli in range(2):
                m = tm & (lbls == cli)
                ax.scatter(emb[m, 0], emb[m, 1],
                           c=COND_COLORS[cond], marker=markers[cli],
                           alpha=0.45, s=18, rasterized=True)
        ax.set_title(f'{mname}  purity={purity:.3f}', fontsize=11)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        results.append(dict(method=mname, purity=round(purity, 4)))

    plt.tight_layout()
    out = os.path.join(out_dir, 'umap_clustering.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    print('Clustering purity (full diff-ERP):')
    for r in results:
        print(f"  {r['method']:14}  purity={r['purity']:.3f}")
    return results

# statistics CSV
def compute_stats(df, out_dir):
    rows = []
    for (ds, mod, font), grp in df.groupby(['dataset', 'modality', 'font_label']):
        par  = grp[grp['condition'] == 'Par'].groupby('subject')['rms_all'].mean()
        ctrl = grp[grp['condition'] == 'Control'].groupby('subject')['rms_all'].mean()
        com  = par.index.intersection(ctrl.index)
        if len(com) < 3:
            continue
        pv, cv = par[com].values, ctrl[com].values
        t, p   = ttest_rel(pv, cv)
        d      = (pv - cv).mean() / ((pv - cv).std(ddof=1) + 1e-12)
        best   = HIGHLIGHT.get((ds, mod, font), False)
        rows.append(dict(
            dataset=ds, modality=mod, font=font,
            n_subjects=len(com),
            mean_rms_par=round(pv.mean(), 4),
            mean_rms_ctrl=round(cv.mean(), 4),
            mean_diff=round((pv - cv).mean(), 4),
            t_stat=round(t, 4), p_value=round(p, 4),
            cohen_d=round(d, 4),
            pct_par_above=round((pv > cv).mean() * 100, 1),
            best_condition=best,
        ))
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(out_dir, 'statistics.csv'), index=False)
    return stats

# main
def main():
    print('=' * 70)
    print('Odd vs Even epoch analysis — Par vs Control')
    print(f'Epoch size: {EPOCH_SAMPLES} samples ({EPOCH_SAMPLES/SR*1000:.1f} ms @ {SR} Hz)')
    print(f'Stimulus rate: {STIM_RATE} Hz; Excluded: {EXCLUDE_SUBJ}')
    print('=' * 70)

    print('\nLoading Angelique')
    rec_ang = build_records(
        ANG_CSV_DIR, parse_angelique_fname, get_files_fn=None,
        dataset_name='Angelique')

    print('\nLoading Talia')
    rec_tal = build_records(
        TAL_CSV_DIR, parse_talia_fname, get_files_fn=_get_talia_files,
        dataset_name='Talia')

    records = rec_ang + rec_tal

    # build flat df for groupby operations
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'diff_erp'}
                       for r in records])

    print(f'\nTotal sequences: {len(df)}')
    print(df.groupby(['dataset', 'modality', 'font_label', 'condition']).size().to_string())

    print('\nStatistics (diff-ERP RMS, all electrodes)')
    stats = compute_stats(df, OUT_DIR)
    for _, r in stats.iterrows():
        sig  = ('***' if r.p_value < 0.001 else '**' if r.p_value < 0.01
                else '*' if r.p_value < 0.05 else 'ns')
        star = ' ★' if r.best_condition else ''
        print(f"  {r.dataset:10} {r.modality:8} {r.font:10}{star} | "
              f"Par={r.mean_rms_par:.4f} Ctrl={r.mean_rms_ctrl:.4f} "
              f"delta={r.mean_diff:+.4f} | t={r.t_stat:+.2f} p={r.p_value:.3f} {sig} "
              f"d={r.cohen_d:+.2f} | {r.pct_par_above:.0f}%")

    print('\nFigures')
    plot_rms_comparison(df, OUT_DIR)
    plot_per_font_rms(df, OUT_DIR)
    plot_diff_erp_waveforms(records, OUT_DIR)

    print('\nUMAP embeddings')
    plot_umap_diff_erp(records, df, OUT_DIR)
    plot_umap_clustered(records, df, OUT_DIR)

    print(f'\nAll outputs -> {OUT_DIR}/')

if __name__ == '__main__':
    main()
