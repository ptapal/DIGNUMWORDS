"""
PRE-SPECIFIED 13-09-2026, before this file was run. Not blind: the internal
reviewer (round 1) already printed TDCA - ROI, RESS - spectral ROI and
TDCA - energy distance from the same CSVs.

QUESTION (research question part 2)
Does a learned operator separate Parity from Control better than the fixed
Retter ROI, within participant?

INPUT (existing CSVs only, nothing is refitted)
time-domain ROI   block_exact_erp_roi.csv / block_exact_erpmean_roi.csv
max-SNR           spatial_exact.csv       learned_diff / learned_mean
CSP               csp2_exact.csv          csp2_diff / csp2_mean
TDCA              tdca_exact.csv          tdca_diff / tdca_mean
RESS              ress_exact.csv          ress_disc / ress_sym
spectral ROI      ress_exact.csv          roi_disc / roi_sym
energy distance   block_exact_erp.csv / block_exact_erpmean.csv
Per-participant z only (z does not depend on the tie rule in
enumerate_stratified, so pre- and post-tie-fix CSVs give identical results).

STATISTIC
Per participant d = z_operator - z_comparator, same arm, paired on subject.
Group: mean d, 95% CI, one-sample t against 0, TWO-sided p (the difference
may go either way), count d > 0.
comparator      time-domain ROI  for max-SNR, CSP, TDCA
                spectral ROI     for RESS (same SNR, same harmonics)

PRIMARY FAMILY
D1, discrimination arm, 4 comparisons (max-SNR, CSP, RESS, TDCA vs ROI),
Holm over the 4 two-sided p.
Secondary, not in the family, reported in full:
- the same 4 in D2 (calibration dataset; Holm shown, descriptive)
- the same 4 on the control arm (mean-ERP; symmetry harmonics for RESS)
- TDCA - energy distance (unsummarised 64 x 68 pattern, no learning),
  both arms, both datasets
- every operator's arms alone (disc, control, disc - control), so the
  sensitivity of each control is visible
- ADDED AFTER THE FIRST RUN (declared, reconciliation only, not in the
  family): the same differences on the paired (disc - ctrl) z. The
  reviewer's M2 table values (TDCA - ROI +1.04 D1, +1.02 D2) match this
  arm, not the discrimination arm the reviewer's R2 spec names.
Sensitivity: the ROI in spatial_exact.csv (roi_diff / roi_mean, recomputed
from the spatial cache) must agree with block_exact_erp_roi.csv; agreement
is printed.

KNOWN MISMATCHES (not fixable from CSVs; R3 addresses the first)
- ROI and max-SNR score a signed RMS amplitude (Parity > Control). TDCA,
  CSP-KL and energy distance are label-swap invariant: they fire on any
  difference, in either direction. So a learned-minus-ROI gain mixes
  spatial learning with a change of scoring.
- CSP is an in-sample covariance divergence with no cross-validation.
- RESS uses the symmetry harmonics as its control, not the mean-ERP.
- z bounds differ: swap-invariant stats max z = sqrt((n-2)(n-1)/(2n))
  (5.52 D1, 2.56 D2); signed stats max z = sqrt((n-1)/2) (5.61, 2.74).

READING
op - ROI > 0 survives Holm in D1   that operator separates the conditions
                                   better than the ROI, under its own
                                   scoring (see mismatches)
ns                                 no evidence of a gain over the ROI
TDCA - energy distance ~ 0         the TDCA advantage is not shown to
                                   come from learning
D2 disagrees with D1               reported; D2 is calibration

    python operator_vs_roi.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as tdist

from block_level_test import OUT_DIR

DS = [('Angelique', 'D1'), ('Talia', 'D2')]


def _z(fn, measure=None):
    d = pd.read_csv(f'{OUT_DIR}/{fn}')
    if measure is not None:
        d = d[d.measure == measure]
    return {ds: g.set_index('subject')['z'] for ds, g in d.groupby('dataset')}


SRC = {
    'ROI (time)':      {'disc': ('block_exact_erp_roi.csv', None),
                        'ctrl': ('block_exact_erpmean_roi.csv', None)},
    'ROI (spectral)':  {'disc': ('ress_exact.csv', 'roi_disc'),
                        'ctrl': ('ress_exact.csv', 'roi_sym')},
    'energy distance': {'disc': ('block_exact_erp.csv', None),
                        'ctrl': ('block_exact_erpmean.csv', None)},
    'max-SNR':         {'disc': ('spatial_exact.csv', 'learned_diff'),
                        'ctrl': ('spatial_exact.csv', 'learned_mean')},
    'CSP':             {'disc': ('csp2_exact.csv', 'csp2_diff'),
                        'ctrl': ('csp2_exact.csv', 'csp2_mean')},
    'RESS':            {'disc': ('ress_exact.csv', 'ress_disc'),
                        'ctrl': ('ress_exact.csv', 'ress_sym')},
    'TDCA':            {'disc': ('tdca_exact.csv', 'tdca_diff'),
                        'ctrl': ('tdca_exact.csv', 'tdca_mean')},
}
FAMILY = [('max-SNR', 'ROI (time)'), ('CSP', 'ROI (time)'),
          ('RESS', 'ROI (spectral)'), ('TDCA', 'ROI (time)')]
EXTRA = [('TDCA', 'energy distance')]


def holm(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m)
    out[o] = np.clip(adj, 0, 1)
    return out


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci,
                t=t, p_two=p, k_pos=int((v > 0).sum()))


def _fmt(label, s, extra=''):
    return (f'{label:34}{s["n"]:>4}{s["mean"]:>+9.3f}'
            f'{f"  [{s["lo"]:+.2f},{s["hi"]:+.2f}]":>18}{s["t"]:>+8.2f}'
            f'{s["p_two"]:>9.4f}{f"{s["k_pos"]}/{s["n"]}":>8}{extra}')


def main():
    Z = {op: {arm: _z(*src) for arm, src in arms.items()} for op, arms in SRC.items()}
    head = (f'{"":34}{"n":>4}{"mean":>9}{"  95% CI":>18}{"t":>8}'
            f'{"p (2t)":>9}{"pos":>8}')
    rows = []

    print('=' * 96)
    print('ARMS ALONE  (per-participant z; positive = Parity/Control separated)')
    print('=' * 96)
    for ds, lab in DS:
        print(f'\n  {lab}\n  {head}')
        for op in SRC:
            a, b = Z[op]['disc'][ds], Z[op]['ctrl'][ds]
            i = a.index.intersection(b.index)
            for arm, v in [('disc', a), ('ctrl', b), ('disc-ctrl', a[i] - b[i])]:
                s = describe(v.values)
                print('    ' + _fmt(f'{op}, {arm}', s))
                rows.append(dict(table='arm', dataset=lab, comparison=op, arm=arm, **s))

    print('\n' + '=' * 96)
    print('OPERATOR - COMPARATOR  (paired per-participant z difference)')
    print('PRIMARY FAMILY: D1 disc, 4 comparisons, Holm')
    print('=' * 96)
    for ds, lab in DS:
        for arm in ('disc', 'ctrl', 'disc-ctrl'):
            print(f'\n  {lab}, {arm} arm\n  {head}{"Holm p":>9}')
            stats = []
            for op, ref in FAMILY + EXTRA:
                if arm == 'disc-ctrl':
                    a = Z[op]['disc'][ds] - Z[op]['ctrl'][ds]
                    b = Z[ref]['disc'][ds] - Z[ref]['ctrl'][ds]
                else:
                    a, b = Z[op][arm][ds], Z[ref][arm][ds]
                i = a.dropna().index.intersection(b.dropna().index)
                stats.append((op, ref, describe((a[i] - b[i]).values)))
            ph = holm([s['p_two'] for _, _, s in stats[:len(FAMILY)]])
            for j, (op, ref, s) in enumerate(stats):
                hp = ph[j] if j < len(FAMILY) else np.nan
                role = ('PRIMARY' if (lab == 'D1' and arm == 'disc' and j < len(FAMILY))
                        else 'secondary')
                print('    ' + _fmt(f'{op} - {ref}', s,
                                    f'{hp:>9.4f}' if j < len(FAMILY) else f'{"(not in family)":>17}'))
                rows.append(dict(table='op_minus_ref', dataset=lab, comparison=f'{op} - {ref}',
                                 arm=arm, p_holm=hp, role=role, **s))

    print('\n' + '=' * 96)
    print('SENSITIVITY: time-domain ROI, block_exact_*_roi.csv vs spatial_exact.csv')
    print('=' * 96)
    for arm, fa, ma in [('disc', 'block_exact_erp_roi.csv', 'roi_diff'),
                        ('ctrl', 'block_exact_erpmean_roi.csv', 'roi_mean')]:
        a, b = _z(fa), _z('spatial_exact.csv', ma)
        for ds, lab in DS:
            i = a[ds].index.intersection(b[ds].index)
            x, y = a[ds][i], b[ds][i]
            print(f'  {lab} {arm}: n={len(i)}  r={np.corrcoef(x, y)[0, 1]:.4f}  '
                  f'max|dz|={np.abs(x - y).max():.4f}  mean {x.mean():+.3f} vs {y.mean():+.3f}')

    out = f'{OUT_DIR}/operator_vs_roi.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
