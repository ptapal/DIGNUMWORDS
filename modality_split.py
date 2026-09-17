"""
PRE-SPECIFIED 14-09-2026, before any result from this file was computed.

QUESTION
Pooling D1 over digits and German number words assumes one effect for both.
Does the discrimination effect hold for each modality separately, and does it
differ between them? Secondary: is D1-digits closer to D2 (digits only)?

READING
both modalities > 0, digits - words n.s.   pooling justified
digits > 0, words n.s., difference sig.    the effect is carried by
                                           digits; pooling dilutes D1
difference n.s. but only digits sig.       underpowered split; no claim
                                           of a modality difference

DEVIATIONS
Calibration median rule is unattainable for swap-invariant operators at
8 assignments: null p takes only {1,3,5,7}/8, so the median is .375 or
.625. Checked instead: the four values each occur 22-29% (uniform), null
mean p .51-.52 for roi_tmpl and tdca in both modalities. Implementation OK.
Ceiling: 18-24 of 30 participants sit at the minimum p (1/8) for roi_tmpl
and tdca in each modality, so z-based digits - words contrasts for those
operators are insensitive and not interpreted. D1-digits vs D2 z are on
different scales (8 vs 16 assignments) and not interpreted either.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind, t as tdist

from csp2 import load_cache
from tdca import prepare
from tdca_ablation import make_stat, _shuffle
from block_level_test import enumerate_stratified, signed_stat, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
from operator_vs_roi import holm

N_SHUF = 20
OPS = ['roi_rms', 'roi_tmpl', 'tdca']


def rms_stat(B):
    v = np.array([np.sqrt((b[:, :, RETTER_ROI_IDX].mean(2).mean(0) ** 2).mean()) for b in B])
    return lambda m: signed_stat(v, m)


def op_stat(B, op):
    if op == 'roi_rms':
        return rms_stat(B)
    return make_stat(B, 'A1_roimean' if op == 'roi_tmpl' else 'A0_tdca')


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t, p=p,
                pos=int((v > 0).sum()))


def fmt(label, s, extra=''):
    return (f'  {label:34}{s["n"]:>4}{s["mean"]:>+8.3f}  [{s["lo"]:+.2f},{s["hi"]:+.2f}]'
            f'{s["t"]:>+7.2f}{s["p"]:>8.4f}{s["pos"]:>4}/{s["n"]}{extra}')


def main():
    meta, D, M = load_cache()
    subsets = {'D1 digits': (meta.dataset == 'Angelique') & (meta.modality == 'Dig'),
               'D1 words': (meta.dataset == 'Angelique') & (meta.modality == 'NumWoGE'),
               'D1 all': meta.dataset == 'Angelique',
               'D2 digits': meta.dataset == 'Talia'}

    rows, cal = [], []
    rng = np.random.default_rng(0)
    for sname, sel in subsets.items():
        for arm, sub in [('disc', D), ('ctrl', M)]:
            prep = prepare(meta[sel], sub, set(EXCLUDE))
            for key, (B, y, st) in sorted(prep.items()):
                for op in OPS:
                    if sname in ('D1 all', 'D2 digits') and op != 'roi_rms':
                        continue # already in tdca_ablation_exact.csv
                    f = op_stat(B, op)
                    _, p, _, z, n_all, _ = enumerate_stratified(f, y, st)
                    rows.append(dict(subset=sname, arm=arm, op=op, subject=key[1], z=z, p=p,
                                     n_assignments=n_all))
                    if arm == 'disc' and sname.startswith('D1 ') and sname != 'D1 all':
                        for _ in range(N_SHUF):
                            cal.append(dict(subset=sname, op=op,
                                            p=enumerate_stratified(f, _shuffle(y, st, rng), st)[1]))
        print(f'done {sname}', flush=True)

    r = pd.DataFrame(rows)
    ab = pd.read_csv(f'{OUT_DIR}/tdca_ablation_exact.csv')
    ab = ab[ab.variant.isin(['A0_tdca', 'A1_roimean'])]
    for sname, ds in [('D1 all', 'Angelique'), ('D2 digits', 'Talia')]:
        g = ab[ab.dataset == ds]
        r = pd.concat([r, pd.DataFrame(dict(subset=sname, arm=g.arm,
                                            op=g.variant.map({'A0_tdca': 'tdca', 'A1_roimean': 'roi_tmpl'}),
                                            subject=g.subject, z=g.z, p=g.p,
                                            n_assignments=g.n_assignments))])
    r.to_csv(f'{OUT_DIR}/modality_split_exact.csv', index=False)

    print('\nCHECKS')
    old = pd.read_csv(f'{OUT_DIR}/block_exact_erp_roi.csv')
    old = old[old.dataset == 'Angelique'].set_index('subject')['z']
    new = r[(r.subset == 'D1 all') & (r.op == 'roi_rms') & (r.arm == 'disc')].set_index('subject')['z']
    i = old.index.intersection(new.index)
    print(f'  roi_rms (cache) vs block_exact_erp_roi, D1 disc: r = {np.corrcoef(old[i], new[i])[0, 1]:.3f}'
          f'   means {new[i].mean():+.3f} vs {old[i].mean():+.3f}')
    c = pd.DataFrame(cal)
    for (s, op), g in c.groupby(['subset', 'op']):
        med = g.p.median()
        print(f'  calibration {s:10} {op:9} null median {med:.3f}  '
              f'{"PASS" if 0.40 <= med <= 0.60 else "FAIL"}')

    out = []
    head = f'  {"":34}{"n":>4}{"mean z":>8}  {"95% CI":^15}{"t":>7}{"p":>8}{"pos":>7}'
    print('\nPER SUBSET (group mean of per-participant z)\n' + head)
    for sname in subsets:
        for arm in ('disc', 'ctrl'):
            for op in OPS:
                v = r[(r.subset == sname) & (r.arm == arm) & (r.op == op)].z
                s = describe(v)
                print(fmt(f'{sname:10} {arm:4} {op}', s))
                out.append(dict(table='subset', subset=sname, arm=arm, contrast=op, **s))
        print()

    print('PRIMARY: digits - words, discrimination arm, paired (Holm over 3 operators)\n' + head)
    res = []
    for op in OPS:
        a = r[(r.subset == 'D1 digits') & (r.arm == 'disc') & (r.op == op)].set_index('subject').z
        b = r[(r.subset == 'D1 words') & (r.arm == 'disc') & (r.op == op)].set_index('subject').z
        i = a.index.intersection(b.index)
        res.append((op, describe(a[i] - b[i])))
    for (op, s), ph in zip(res, holm([s['p'] for _, s in res])):
        print(fmt(f'digits - words  {op}', s, f'   Holm {ph:.4f}'))
        out.append(dict(table='digits_minus_words', subset='D1', arm='disc', contrast=op, p_holm=ph, **s))

    print('\nSECONDARY: operator - roi_rms within subset, discrimination arm\n' + head)
    for sname in subsets:
        base = r[(r.subset == sname) & (r.arm == 'disc') & (r.op == 'roi_rms')].set_index('subject').z
        for op in ('roi_tmpl', 'tdca'):
            a = r[(r.subset == sname) & (r.arm == 'disc') & (r.op == op)].set_index('subject').z
            i = a.index.intersection(base.index)
            s = describe(a[i] - base[i])
            print(fmt(f'{sname:10} {op} - roi_rms', s))
            out.append(dict(table='op_minus_roi', subset=sname, arm='disc', contrast=f'{op} - roi_rms', **s))

    print('\nSECONDARY: D1 digits vs D2 digits, discrimination arm (Welch; z scales differ: 8 vs 16 assignments)')
    for op in OPS:
        a = r[(r.subset == 'D1 digits') & (r.arm == 'disc') & (r.op == op)].z
        b = r[(r.subset == 'D2 digits') & (r.arm == 'disc') & (r.op == op)].z
        t, p = ttest_ind(a, b, equal_var=False)
        print(f'  {op:9} D1 digits {a.mean():+.3f}  D2 {b.mean():+.3f}  t={t:+.2f} p={p:.4f}')
        out.append(dict(table='d1dig_vs_d2', subset='D1dig-D2', arm='disc', contrast=op,
                        mean=a.mean() - b.mean(), t=t, p=p))
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/modality_split_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/modality_split_exact.csv, modality_split_summary.csv')


if __name__ == '__main__':
    main()
