import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel, ttest_1samp, wilcoxon, binomtest, t as tdist

from electrodes import RETTER_ROI

ROOT   = os.path.dirname(os.path.abspath(__file__))  
RES    = os.path.join(ROOT, 'results', 'within_subject')
H5     = os.path.join(ROOT, 'h5_new', 'talia_seqlevel.h5')
DS     = [('Angelique', 'D1'), ('Talia', 'D2')]

# order features so the primary result comes first
ORDER = ['erp', 'erp_all', 'erpmean', 'erp_roi', 'erpmean_roi', 'erp_ratio_roi',
         'snr', 'snr_roi', 'snr_ratio_roi']
ELEC  = ['erp_elec', 'erpmean_elec', 'erp_ratio_elec',
         'snr_elec', 'snr_ratio_elec']
# (discrimination measure, matched general-response control)
PAIRS = [('erp',      'erpmean',      'exact', 'Full pattern'),
         ('erp_roi',  'erpmean_roi',  'exact', 'ROI amplitude'),
         ('erp_elec', 'erpmean_elec', 'elec',  'ROI enrichment')]


def _stale(path):
    return os.path.exists(H5) and os.path.getmtime(path) < os.path.getmtime(H5)


def _read(kind, feature):
    """kind: 'exact' | 'elec'. Returns DataFrame or None."""
    fn = os.path.join(RES, f'block_{"exact" if kind == "exact" else "localize"}'
                           f'_{feature}.csv')
    if not os.path.exists(fn):
        return None
    if _stale(fn):
        print(f'  !! STALE (predates the D2 fix): {os.path.basename(fn)}')
    return pd.read_csv(fn)


def _subject_z(feature):
    """Per-subject effect size, indexed by subject, split by dataset."""
    d = _read('exact', feature)
    if d is None:
        return None
    return {ds: g.set_index('subject')['z'] for ds, g in d.groupby('dataset')}


def _roi_minus_rest(feature):
    """Per-subject ROI enrichment, indexed by subject, split by dataset."""
    d = _read('elec', feature)
    if d is None:
        return None
    out = {}
    for ds, g in d.groupby('dataset'):
        piv = g.pivot(index='subject', columns='elec', values='z')
        missing = [c for c in RETTER_ROI if c not in piv.columns]
        if missing:
            raise KeyError(f'{feature}: ROI channels absent: {missing}')
        out[ds] = (piv[RETTER_ROI].mean(axis=1)
                   - piv.drop(columns=RETTER_ROI).mean(axis=1))
    return out


def _describe(v, one_sided=True):
    """Mean, CI, t, sign test for a vector of per-subject values."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    ci = tdist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    k = int((v > 0).sum())
    bt = binomtest(k, n, 0.5, alternative='greater').pvalue
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci,
                t=t, p=(p / 2 if one_sided else p), k=k, sign_p=bt)


def table_group():
    print('\n' + '=' * 96)
    print('GROUP LEVEL --- per-subject effect size z (own permutation distribution)')
    print('=' * 96)
    print(f'{"feature":16}{"ds":4}{"n":>4}{"z mean":>9}{"  95% CI":>18}'
          f'{"t":>8}{"Stouffer":>10}{"p":>11}{"pos":>8}{"sign p":>9}')
    print('-' * 96)
    rows = []
    for f in ORDER:
        d = _read('exact', f)
        if d is None:
            continue
        for ds, lab in DS:
            g = d[d.dataset == ds]
            if g.empty:
                continue
            s = _describe(g['z'].values)
            pm = np.clip(g['p'].values, 1e-9, 1 - 1e-9)
            zs = norm.isf(pm).sum() / np.sqrt(len(pm))
            print(f'{f:16}{lab:4}{s["n"]:>4}{s["mean"]:>+9.3f}'
                  f'{f"  [{s['lo']:+.2f},{s['hi']:+.2f}]":>18}'
                  f'{s["t"]:>+8.2f}{zs:>+10.2f}{norm.sf(zs):>11.2e}'
                  f'{f"{s['k']}/{s['n']}":>8}{s["sign_p"]:>9.4f}')
            rows.append(dict(table='group', feature=f, dataset=lab,
                             stouffer_z=zs, stouffer_p=norm.sf(zs), **s))
    return pd.DataFrame(rows)


def table_roi():
    print('\n' + '=' * 96)
    print('ROI ENRICHMENT  --- mean z over 8 Retter channels minus mean z over other 60')
    print('=' * 96)
    print(f'{"map":18}{"ds":4}{"n":>4}{"ROI-rest":>10}{"  95% CI":>18}'
          f'{"t":>8}{"p (1t)":>9}{"Wilcox":>9}{"pos":>8}')
    print('-' * 96)
    rows = []
    for f in ELEC:
        got = _roi_minus_rest(f)
        if got is None:
            continue
        for ds, lab in DS:
            if ds not in got:
                continue
            v = got[ds].values
            s = _describe(v)
            _, pw = wilcoxon(v, alternative='greater')
            print(f'{f:18}{lab:4}{s["n"]:>4}{s["mean"]:>+10.3f}'
                  f'{f"  [{s['lo']:+.2f},{s['hi']:+.2f}]":>18}'
                  f'{s["t"]:>+8.2f}{s["p"]:>9.4f}{pw:>9.4f}'
                  f'{f"{s['k']}/{s['n']}":>8}')
            rows.append(dict(table='roi', feature=f, dataset=lab,
                             wilcoxon_p=pw, **s))
    return pd.DataFrame(rows)


def table_signal_vs_control():
    print('\n' + '=' * 96)
    print('SIGNAL vs CONTROL  --- discrimination minus general response, paired')
    print('   (same epochs, same blocks: gain, coverage and session state cancel)')
    print('=' * 96)
    print(f'{"comparison":22}{"ds":4}{"n":>4}{"sig-ctl":>10}{"  95% CI":>18}'
          f'{"t":>8}{"p (1t)":>9}{"Wilcox":>9}{"pos":>8}')
    print('-' * 96)
    rows = []
    for sig_f, ctl_f, kind, label in PAIRS:
        get = _subject_z if kind == 'exact' else _roi_minus_rest
        a_all, b_all = get(sig_f), get(ctl_f)
        if a_all is None or b_all is None:
            print(f'{label:22} skipped (missing {sig_f} or {ctl_f})')
            continue
        for ds, lab in DS:
            if ds not in a_all or ds not in b_all:
                continue
            a, b = a_all[ds], b_all[ds]
            idx = a.index.intersection(b.index)
            v = (a[idx] - b[idx]).values
            s = _describe(v)
            t, p = ttest_rel(a[idx], b[idx])
            _, pw = wilcoxon(v, alternative='greater')
            print(f'{label:22}{lab:4}{s["n"]:>4}{s["mean"]:>+10.3f}'
                  f'{f"  [{s['lo']:+.2f},{s['hi']:+.2f}]":>18}'
                  f'{t:>+8.2f}{p/2:>9.4f}{pw:>9.4f}'
                  f'{f"{s['k']}/{s['n']}":>8}')
            rows.append(dict(table='signal_vs_control', feature=label,
                             dataset=lab, t=t, p=p / 2, wilcoxon_p=pw,
                             n=s['n'], mean=s['mean'], lo=s['lo'], hi=s['hi'],
                             k=s['k'], sign_p=s['sign_p']))
    return pd.DataFrame(rows)


def main():
    if not os.path.isdir(RES):
        sys.exit(f'no results directory at {RES}')
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RES, 'block_*.csv')))
    print(f'{len(have)} result files in {os.path.relpath(RES, ROOT)}')

    g = table_group()
    r = table_roi()
    s = table_signal_vs_control()

    if '--csv' in sys.argv:
        out = pd.concat([g, r, s], ignore_index=True)
        fn = os.path.join(RES, 'summary_tables.csv')
        out.to_csv(fn, index=False)
        print(f'\nwrote {os.path.relpath(fn, ROOT)}')


if __name__ == '__main__':
    main()
