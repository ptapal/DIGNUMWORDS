"""
USAGE:
python block_level_test.py erp
python block_level_test.py erp --all             # keep S20/S30
python block_level_test.py erp --drop=S02,S03    # sensitivity analysis
"""
import sys
import numpy as np
import pandas as pd
from itertools import combinations, product
from scipy.stats import norm, wilcoxon, binomtest, t as tdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rebuild_seqlevel import (load_seqlevel_h5, build_df, ANG_H5_OUT, TAL_H5_OUT,
                              RANDOM, N_FEATS, parse_angelique_fname,
                              parse_talia_fname, _get_talia_files)
from electrodes import biosemi_68_order, RETTER_ROI, RETTER_ROI_IDX

OUT_DIR  = 'results/within_subject'
EXCLUDE  = {'S20', 'S30'}
DS_LABEL = {'Angelique': 'D1', 'Talia': 'D2'}
MAX_PERM = 20000     

SNR_DISC  = 0    # column of FEAT_NAMES holding SNR @ 3.75 Hz
SNR_RATIO = 10   # column holding SNR(3.75) / SNR(7.5)


# feature specs
def _snr_mat(flat):
    """(748,) -> (68 electrodes, 11 features)."""
    return np.asarray(flat, dtype=np.float64).reshape(-1, N_FEATS)


def _rms(erp2d, cols=None):
    """RMS over time (and over `cols` electrodes if given)."""
    x = erp2d if cols is None else erp2d[:, cols]
    return float(np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean()))


def _rms_per_elec(erp2d):
    """(time, 68) -> (68,) RMS over time at each electrode."""
    return np.sqrt((np.asarray(erp2d, dtype=np.float64) ** 2).mean(axis=0))


# each extractor maps one sequence to a 1-D vector
EXTRACT_SNR = {
    'snr_full':       lambda f: np.asarray(f, dtype=np.float64),
    'snr_roi':        lambda f: np.array([_snr_mat(f)[RETTER_ROI_IDX, SNR_DISC].mean()]),
    'snr_ratio_roi':  lambda f: np.array([_snr_mat(f)[RETTER_ROI_IDX, SNR_RATIO].mean()]),
    'snr_elec':       lambda f: _snr_mat(f)[:, SNR_DISC],
    'snr_ratio_elec': lambda f: _snr_mat(f)[:, SNR_RATIO],
}

EXTRACT_ERP = {
    'erp_full':       lambda r: r['diff_erp'].ravel().astype(np.float64),
    'erpmean_full':   lambda r: r['mean_erp'].ravel().astype(np.float64),
    'erp_roi':        lambda r: np.array([_rms(r['diff_erp'], RETTER_ROI_IDX)]),
    'erpmean_roi':    lambda r: np.array([_rms(r['mean_erp'], RETTER_ROI_IDX)]),
    'erp_ratio_roi':  lambda r: np.array([_rms(r['diff_erp'], RETTER_ROI_IDX) /
                                          (_rms(r['mean_erp'], RETTER_ROI_IDX) + 1e-12)]),
    'erp_elec':       lambda r: _rms_per_elec(r['diff_erp']),
    'erpmean_elec':   lambda r: _rms_per_elec(r['mean_erp']),
    'erp_ratio_elec': lambda r: _rms_per_elec(r['diff_erp']) /
                                (_rms_per_elec(r['mean_erp']) + 1e-12),
}

FEATURES = {
    'snr':            dict(source='snr', extract='snr_full',       stat='energy'),
    'snr_roi':        dict(source='snr', extract='snr_roi',        stat='signed'),
    'snr_ratio_roi':  dict(source='snr', extract='snr_ratio_roi',  stat='signed'),
    'snr_elec':       dict(source='snr', extract='snr_elec',       stat='localize'),
    'snr_ratio_elec': dict(source='snr', extract='snr_ratio_elec', stat='localize'),
    'erp':            dict(source='erp', extract='erp_full',       stat='energy'),
    'erp_roi':        dict(source='erp', extract='erp_roi',        stat='signed'),
    'erp_ratio_roi':  dict(source='erp', extract='erp_ratio_roi',  stat='signed'),
    'erp_elec':       dict(source='erp', extract='erp_elec',       stat='localize'),
    'erp_ratio_elec': dict(source='erp', extract='erp_ratio_elec', stat='localize'),
    'erpmean':        dict(source='erp', extract='erpmean_full',   stat='energy'),
    'erpmean_roi':    dict(source='erp', extract='erpmean_roi',    stat='signed'),
    'erpmean_elec':   dict(source='erp', extract='erpmean_elec',   stat='localize'),
}


# stats
def energy_stat(X, mask):
    """Energy distance between X[mask] and X[~mask]. Larger = more separated.
    Unsigned: fires on any difference, in any direction.
    """
    A, B = X[mask], X[~mask]
    dab = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).mean()
    daa = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2).mean()
    dbb = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2).mean()
    return 2 * dab - daa - dbb


def signed_stat(v, mask):
    """mean(Parity) - mean(Control). Positive = the predicted direction."""
    return v[mask].mean() - v[~mask].mean()


# permutation core
def _assign(combo, n):
    mask = np.zeros(n, dtype=bool)
    for idx, sel in combo:
        mask[idx[sel]] = True
    return mask


def enumerate_stratified(stat_fn, y, strata, seed=RANDOM):
    """Enumerate every condition assignment that preserves the block design.

    Labels are permuted only among blocks sharing a stratum (font x modality),
    which is the randomisation the experiment actually performed.

    Returns (observed, mid_p_upper, p_two, z, n_assignments, exact).

    mid_p splits ties, so it cannot reach 0 and the attainable minimum is
    0.5 / n_assignments -- the design's resolution ceiling.
    """
    per = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        k   = int(y[idx].sum())
        per.append([(idx, np.isin(np.arange(len(idx)), c))
                    for c in combinations(range(len(idx)), k)])
    n_all = int(np.prod([len(p) for p in per]))
    n     = len(y)

    if n_all <= MAX_PERM:
        stats = np.array([stat_fn(_assign(c, n)) for c in product(*per)])
        exact = True
    else:
        rng   = np.random.default_rng(seed)
        stats = np.array([
            stat_fn(_assign([opts[rng.integers(len(opts))] for opts in per], n))
            for _ in range(MAX_PERM)])
        exact = False

    obs   = stat_fn(y.astype(bool))
    mid_p = (np.sum(stats > obs) + 0.5 * np.sum(stats == obs)) / len(stats)
    p_two = min(1.0, 2 * min(mid_p, 1 - mid_p))
    z     = (obs - stats.mean()) / (stats.std(ddof=1) + 1e-12)
    return float(obs), float(mid_p), float(p_two), float(z), n_all, exact


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate(
        (p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


# data load
def load_blocks(feature, exclude):
    """Average sequences within each source file into one vector per block.

    Returns {(dataset, subject): (X, y, strata)} where
      X      (n_blocks, d)  block feature vectors (d == 1 for scalar features)
      y      (n_blocks,)    1 = Parity, 0 = Control
      strata (n_blocks,)    the font x modality experimental block
    """
    spec = FEATURES[feature]

    if spec['source'] == 'snr':
        rec = load_seqlevel_h5(ANG_H5_OUT, 'Angelique') + \
              load_seqlevel_h5(TAL_H5_OUT, 'Talia')
        df  = build_df(rec)
        df  = df[~df['subject'].isin(exclude)].reset_index(drop=True)
        fn  = EXTRACT_SNR[spec['extract']]
        df['vec'] = [fn(f) for f in df['flat'].values]
    else:
        from epoch_analysis import build_records, ANG_CSV_DIR, TAL_CSV_DIR
        recs = build_records(ANG_CSV_DIR, parse_angelique_fname,
                             get_files_fn=None, dataset_name='Angelique',
                             exclude_subj=exclude) + \
               build_records(TAL_CSV_DIR, parse_talia_fname,
                             get_files_fn=_get_talia_files, dataset_name='Talia',
                             exclude_subj=exclude)
        df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ('diff_erp', 'mean_erp')} for r in recs])
        fn = EXTRACT_ERP[spec['extract']]
        df['vec'] = [fn(r) for r in recs]

    out = {}
    for (ds, subj), sub in df.groupby(['dataset', 'subject']):
        rows, labs, strata = [], [], []
        for _, g in sub.groupby('block_id'):
            conds = set(g['condition'])
            assert len(conds) == 1, f'{ds}/{subj}: a block spans {conds}'
            rows.append(np.stack(g['vec'].values).mean(axis=0))
            labs.append(1 if g['condition'].iloc[0] == 'Par' else 0)
            strata.append(f"{g['modality'].iloc[0]}|{g['font_label'].iloc[0]}")
        out[(ds, subj)] = (np.stack(rows), np.array(labs), np.array(strata))
    return out


# localisation
def run_localize(blocks, feature, tag):
    """Signed Parity — Control statistic at every electrode, same stratified
    null. Replaces the descriptive topographies with tested ones."""
    rows = []
    for (ds, subj), (Xb, y, strata) in sorted(blocks.items()):
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            continue
        for e in range(Xb.shape[1]):
            _, p, _, z, _, _ = enumerate_stratified(
                lambda m, v=Xb[:, e]: signed_stat(v, m), y, strata)
            rows.append(dict(dataset=ds, subject=subj, elec_idx=e,
                             elec=biosemi_68_order[e],
                             z=round(z, 4), p=round(p, 5)))

    r   = pd.DataFrame(rows)
    out = f'{OUT_DIR}/block_localize_{feature}{tag}.csv'
    r.to_csv(out, index=False)

    print(f'\n{"=" * 70}')
    print(f'PER-ELECTRODE LOCALISATION — {feature}')
    print(f'positive z = Parity > Control at that electrode')
    print(f'{"=" * 70}')

    summaries = []
    for ds, g in r.groupby('dataset'):
        piv = g.pivot(index='subject', columns='elec', values='z')
        n   = len(piv)
        t   = piv.mean() / (piv.std(ddof=1) / np.sqrt(n))
        pv  = 2 * tdist.sf(np.abs(t), n - 1)
        res = pd.DataFrame({'dataset': ds, 'elec': piv.columns,
                            'z': piv.mean().values, 't': t.values,
                            'p': pv, 'q': bh_fdr(pv)})
        res['elec_idx'] = [biosemi_68_order.index(e) for e in res['elec']]
        summaries.append(res)

        roi = res.set_index('elec').loc[RETTER_ROI]
        print(f'\n  {DS_LABEL.get(ds, ds)}  (n={n} subjects, 68 electrodes, BH-FDR)')
        print(f'    electrodes with q<.05:            {(res.q < .05).sum()}/68')
        print(f'    of the 8 Retter ROI channels:     {(roi.q < .05).sum()}/8')
        print(f'    mean z over ROI = {roi.z.mean():+.3f}   '
              f'mean z over the other 60 = '
              f'{res[~res.elec.isin(RETTER_ROI)].z.mean():+.3f}')
        print('\n' + res.sort_values('t', ascending=False)
                        .head(12)[['elec', 'z', 't', 'p', 'q']]
                        .round(4).to_string(index=False))

    summ = pd.concat(summaries, ignore_index=True).sort_values(['dataset', 'elec_idx'])
    summ_csv = f'{OUT_DIR}/block_localize_{feature}{tag}_summary.csv'
    summ.to_csv(summ_csv, index=False)
    print(f'\nwrote {out}')
    print(f'wrote {summ_csv}  (per-electrode z -> feed to mne.viz.plot_topomap)')


def parse_args(argv):
    pos = [a for a in argv[1:] if not a.startswith('--')]
    feature = pos[0] if pos else 'snr'
    if feature not in FEATURES:
        sys.exit(f'unknown feature {feature!r}\nchoose from:\n  ' +
                 '\n  '.join(FEATURES))

    exclude = set() if '--all' in argv else set(EXCLUDE)
    dropped = False
    for a in argv:
        if a.startswith('--drop='):
            exclude |= {s.strip() for s in a.split('=', 1)[1].split(',') if s.strip()}
            dropped = True
    tag = ('_all' if '--all' in argv else '') + ('_drop' if dropped else '')
    return feature, exclude, tag


def main():
    feature, exclude, tag = parse_args(sys.argv)
    spec   = FEATURES[feature]
    signed = spec['stat'] in ('signed', 'ratio')   # both are scalar per block

    print(f'feature={feature}  statistic={spec["stat"]}  '
          f'excluded={sorted(exclude) or "none"}')
    if signed or spec['stat'] == 'localize':
        print(f'ROI = {", ".join(RETTER_ROI)}')

    blocks = load_blocks(feature, exclude)

    if spec['stat'] == 'localize':
        run_localize(blocks, feature, tag)
        return

    rows, skipped = [], []
    for (ds, subj), (Xb, y, strata) in sorted(blocks.items()):
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            skipped.append(f'{ds}/{subj}')
            continue

        if signed:
            stat_fn = lambda m, v=Xb.ravel(): signed_stat(v, m)
        else:
            Xs = StandardScaler().fit_transform(Xb.astype(np.float64))
            Xs = PCA(n_components=min(len(y) - 1, Xs.shape[1]),
                     random_state=RANDOM).fit_transform(Xs)
            stat_fn = lambda m, Xs=Xs: energy_stat(Xs, m)

        stat, p, p_two, z, n_all, exact = enumerate_stratified(stat_fn, y, strata)
        # a stratum holding only one condition cannot be permuted;
        # it contributes a factor of 1 and lowers that subject's resolution
        degen = sum(1 for s in np.unique(strata) if len(set(y[strata == s])) < 2)
        rows.append(dict(dataset=ds, subject=subj, n_blocks=len(y),
                         n_par=int(y.sum()), n_strata=len(np.unique(strata)),
                         degenerate_strata=degen, stat=round(stat, 5),
                         z=round(z, 4), p=round(p, 5), p_two=round(p_two, 5),
                         n_assignments=n_all, exact=exact))

    if skipped:
        print(f'skipped (too few blocks per condition): {", ".join(skipped)}')

    r = pd.DataFrame(rows)
    r['q'] = np.nan
    for ds, g in r.groupby('dataset'):
        r.loc[g.index, 'q'] = bh_fdr(g['p'].values)
    out_csv = f'{OUT_DIR}/block_exact_{feature}{tag}.csv'
    r.to_csv(out_csv, index=False)

    print(f'\n{"=" * 70}')
    print(f'EXACT BLOCK-LEVEL RANDOMISATION TEST — {feature}')
    print(f'{"=" * 70}')

    for ds, g in r.groupby('dataset'):
        m      = len(g)
        # best case across subjects: most assignments -> smallest attainable p.
        # If even that cannot clear BH, no subject can reach significance.
        floor  = 0.5 / int(g.n_assignments.max())
        needed = 0.05 / m

        print(f'\n  {DS_LABEL.get(ds, ds)}  n={m} subjects, '
              f'blocks/subject={[int(v) for v in sorted(g.n_blocks.unique())]}, '
              f'strata/subject={[int(v) for v in sorted(g.n_strata.unique())]}')
        print(f'    enumeration: {[int(v) for v in sorted(g.n_assignments.unique())]} '
              f'assignments {"(exact)" if g.exact.all() else "(sampled)"}')
        print(f'    design ceiling: best attainable p = {floor:.4f}; '
              f'BH q<.05 over {m} subjects needs p <= {needed:.4f}  '
              f'-> individual-level inference '
              f'{"POSSIBLE" if floor <= needed else "NOT ATTAINABLE"}')
        if (g.degenerate_strata > 0).any():
            bad = g.loc[g.degenerate_strata > 0, 'subject'].tolist()
            print(f'    reduced resolution (incomplete stratum): {", ".join(bad)}')
        print(f'    subjects p<.05 uncorrected: {(g.p < .05).sum()}/{m} '
              f'(null expectation ~{0.05 * m:.1f})')
        print(f'    subjects q<.05 (BH-FDR):    {(g.q < .05).sum()}/{m}')
        print(f'    median p = {g.p.median():.3f}  (0.500 under the null)')

    print(f'\n  GROUP LEVEL  (per-subject mid-p is uniform under the null)')
    if signed:
        print(f'  positive z = Parity > Control, the predicted direction')
    for ds, g in r.groupby('dataset'):
        pm    = np.clip(g['p'].values, 1e-9, 1 - 1e-9)
        zs    = norm.isf(pm).sum() / np.sqrt(len(pm))
        d     = pm - 0.5
        pw    = wilcoxon(d, alternative='less').pvalue if np.any(d != 0) else np.nan
        zz    = g['z'].values
        ci    = tdist.ppf(0.975, len(zz) - 1) * zz.std(ddof=1) / np.sqrt(len(zz))
        tstat = zz.mean() / (zz.std(ddof=1) / np.sqrt(len(zz)))
        k     = int((zz > 0).sum())
        bt    = binomtest(k, len(zz), 0.5, alternative='greater').pvalue

        print(f'\n    {DS_LABEL.get(ds, ds)}  (n={len(g)})')
        print(f'      Stouffer z = {zs:+.2f}   p = {norm.sf(zs):.2e}')
        print(f'      Wilcoxon (mid-p < .5)    p = {pw:.4f}')
        print(f'      per-subject z: M = {zz.mean():+.3f}  '
              f'95% CI [{zz.mean() - ci:+.3f}, {zz.mean() + ci:+.3f}]  '
              f't({len(zz) - 1}) = {tstat:+.2f}')
        print(f'      {k}/{len(zz)} subjects with z > 0   sign test p = {bt:.2e}')

    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
