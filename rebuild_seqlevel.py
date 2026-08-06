import os
import re
import sys
import gc
import h5py
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from electrodes import biosemi_68_order, region_map

# paths
ANG_CSV_DIR  = 'data/angelique_v2/unlabaled_data_angelique_v2'
TAL_CSV_DIR  = 'data/talia/unlabeled_data_talia'
ANG_H5_OUT   = 'h5_new/angelique_v2_seqlevel.h5'
TAL_H5_OUT   = 'h5_new/talia_seqlevel.h5'
OUT_DIR      = 'results/seqlevel'
os.makedirs(OUT_DIR, exist_ok=True)

# constants
SR         = 512          # sampling rate
N_FFT      = 32768        # zero-pad to this (2^15); 3.75 Hz = bin 240 exactly
N_ELEC     = 68           # EEG electrodes (drop EOG)
N_NOISE    = 20           # noise bins on each side when computing SNR
RANDOM     = 42

FREQS_TARGET = (3.75, 7.5, 11.25, 15.0, 18.75)
TARGET_BINS  = tuple(int(round(f * N_FFT / SR)) for f in FREQS_TARGET)
# -> (240, 480, 720, 960, 1200)

FEAT_NAMES = [
    'SNR_3.75', 'SNR_7.5', 'SNR_11.25', 'SNR_15', 'SNR_18.75',
    'logP_3.75', 'logP_7.5', 'logP_11.25', 'logP_15', 'logP_18.75',
    'SNR_ratio',
]
N_FEATS    = len(FEAT_NAMES)  # 11
N_FLAT     = N_ELEC * N_FEATS # 748

ELEC_NAMES = biosemi_68_order[:N_ELEC]
OCC_KEYS   = ['Occipital', 'Parietooccipital']
OCC_IDX    = sorted([ELEC_NAMES.index(e)
                     for k in OCC_KEYS for e in region_map[k]
                     if e in ELEC_NAMES])

COND_COLORS = {'Par': '#D6604D', 'Control': '#4393C3'}
UMAP_PARAMS = dict(n_neighbors=12, min_dist=0.1, random_state=RANDOM)


# core feature computation
def compute_seq_features(eeg):
    N = eeg.shape[0]
    n_elec = min(eeg.shape[1], N_ELEC)

    # Hanning window to reduce spectral leakage
    win   = np.hanning(N)
    eeg_w = (eeg[:, :n_elec] * win[:, np.newaxis]).astype(np.float64)

    # FFT -> PSD (zero-padded to N_FFT)
    fft_vals = np.fft.rfft(eeg_w, n=N_FFT, axis=0) # (N_FFT//2+1, n_elec)
    psd      = np.abs(fft_vals) ** 2               # power spectral density

    target_bin_set = set(TARGET_BINS)
    n_t = len(FREQS_TARGET)
    feat = np.zeros((N_ELEC, N_FEATS), dtype=np.float32)

    for fi, tbin in enumerate(TARGET_BINS):
        noise_bins = [b for b in range(tbin - N_NOISE, tbin + N_NOISE + 1)
                      if b != tbin and 0 <= b < psd.shape[0]
                      and b not in target_bin_set]
        noise_psd  = psd[noise_bins, :n_elec].mean(axis=0)
        signal_psd = psd[tbin, :n_elec]

        feat[:, fi]       = (signal_psd / (noise_psd + 1e-30)).astype(np.float32)
        feat[:, n_t + fi] = np.log10(signal_psd + 1e-30).astype(np.float32)

    feat[:, -1] = feat[:, 0] / (feat[:, 1] + 1e-10) # SNR_ratio
    return feat

# CSV loaders
def _pivot_csv(fpath, n_elec=N_ELEC):
    """
    Load a raw EEG CSV and return dict {condition_idx: eeg (N, n_elec)}.
    CSV columns: time, channel, condition, value
    """
    df = pd.read_csv(fpath, dtype={'time': np.int32, 'channel': np.int8,
                                   'condition': np.int8, 'value': np.float32})
    result = {}
    for cond_idx in sorted(df['condition'].unique()):
        sub = df[df['condition'] == cond_idx]
        n_times = sub['time'].nunique()
        # pivot: rows=time, cols=channel
        pivoted = (sub.pivot(index='time', columns='channel', values='value')
                     .values[:, :n_elec]) # (N, n_elec)
        result[int(cond_idx)] = pivoted.astype(np.float32)
    del df; gc.collect()
    return result

def _load_talia_csv(fpath, n_elec=N_ELEC):
    """
    D2 export bug: the `value` column is laid out (sequence, channel, time)
    while the time/channel/condition label columns were written for
    (time, channel, sequence). Pivoting on the labels therefore reads samples
    out of order -- lag-1 autocorrelation 0.10 instead of 0.99 -- which whitens
    the spectrum and destroys the periodic response entirely (SNR at 7.5 Hz
    over the occipitotemporal ROI: 0.88 mislabelled vs ~4 reconstructed).
    Reconstruct from the raw value order instead of the labels.
    """
    df = pd.read_csv(fpath, dtype={'time': np.int32, 'channel': np.int8,
                                   'condition': np.int8, 'value': np.float32})
    n_seq  = df['condition'].nunique()
    n_ch   = df['channel'].nunique()
    n_time = df['time'].nunique()
    if n_seq * n_ch * n_time != len(df):
        raise ValueError(
            f'{os.path.basename(fpath)}: {len(df)} rows is not '
            f'{n_seq} seq x {n_ch} ch x {n_time} time -- layout unknown')

    arr = df['value'].values.reshape(n_seq, n_ch, n_time)
    del df; gc.collect()
    return {s: np.ascontiguousarray(arr[s].T)[:, :n_elec].astype(np.float32)
            for s in range(n_seq)}


def _check_temporal_order(fpath, seq_data, min_ac=0.8):
    """EEG at 512 Hz has lag-1 autocorrelation ~0.99. Anything near zero means
    the samples are not in temporal order, which silently destroys every
    frequency-domain and epoch-based measure downstream."""
    for s, eeg in seq_data.items():
        if eeg.shape[0] < 3:
            continue
        x = eeg[:, 0].astype(np.float64)
        ac = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if not np.isfinite(ac) or ac < min_ac:
            raise ValueError(
                f'{os.path.basename(fpath)} seq {s}: lag-1 autocorrelation '
                f'{ac:.3f} < {min_ac} -- samples are not in temporal order')


def load_sequences(fpath, dataset_name, n_elec=N_ELEC):
    """Load one epbin CSV into {sequence_index: (n_time, n_elec)}.

    Dispatches on dataset because the two exports use different value
    layouts (see _load_talia_csv), then verifies temporal ordering.
    """
    if dataset_name == 'Talia':
        seq_data = _load_talia_csv(fpath, n_elec=n_elec)
    else:
        seq_data = _pivot_csv(fpath, n_elec=n_elec)
    _check_temporal_order(fpath, seq_data)
    return seq_data


def parse_angelique_fname(fname):
    """
    Returns (subject, modality, font_type, specific_font, condition)
    subject is zero-padded string like '02'.
    """
    base = os.path.basename(fname).replace('.csv', '')
    sm   = re.search(r'S(\d+)$', base)
    subj = sm.group(1).zfill(2) if sm else 'Unknown'

    parts = base.split('_')
    try:
        idx = parts.index('epbin') + 1
    except ValueError:
        idx = 1

    category  = parts[idx]     if len(parts) > idx     else None
    font_type = parts[idx + 1] if len(parts) > idx + 1 else None
    part3     = parts[idx + 2] if len(parts) > idx + 2 else ''

    SFMAP = {'Atp': 'A', 'Std': 'S', 'A': 'A', 'S': 'S'}
    if part3 in SFMAP:
        specific_font  = SFMAP[part3]
        cond_part      = parts[idx + 3] if len(parts) > idx + 3 else ''
    else:
        specific_font  = None
        cond_part      = part3

    cond = 'Par' if cond_part.startswith('Par') else 'Control'
    return subj, category, font_type, specific_font, cond

def parse_talia_fname(fname):
    """
    Returns (subject, modality, font_type, specific_font, condition)
    epb_{X}{Y}_... where X=1->Par, X=2->Control, Y=font code
    """
    base = os.path.basename(fname).replace('.csv', '')
    sm   = re.search(r'S(\d+)(?:_icf)?$', base)
    subj = sm.group(1).zfill(2) if sm else 'Unknown'

    cm = re.match(r'epb_(\d)(\d)', base)
    par_code  = cm.group(1) if cm else '1'
    font_code = cm.group(2) if cm else '1'

    cond = 'Par' if par_code == '1' else 'Control'
    font_map = {
        '1': ('1F',  None),
        '2': ('10F', 'S'),
        '3': ('10F', 'M'),
        '4': ('20F', 'HD'),
    }
    font_type, specific_font = font_map.get(font_code, ('1F', None))
    return subj, 'Dig', font_type, specific_font, cond


def _get_talia_files(csv_dir):
    """Return one file per (subject, condition_code), preferring _icf."""
    all_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    best = {}
    for f in all_files:
        m = re.match(r'(epb_\d\d)_.*_(S\d+)(_icf)?\.csv$', f)
        if not m:
            continue
        key     = (m.group(2), m.group(1)) # (subject, condition_prefix)
        is_icf  = m.group(3) is not None
        if key not in best or is_icf:
            best[key] = f
    return list(best.values())

# build H5
def build_h5(csv_dir, h5_out, parse_fn, get_files_fn=None, dataset_name='?',
             n_elec=N_ELEC):
    if os.path.exists(h5_out):
        print(f'[{dataset_name}] H5 already exists at {h5_out} — skipping build.')
        print('  Delete it to rebuild.')
        return

    if get_files_fn:
        files = get_files_fn(csv_dir)
    else:
        files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    print(f'[{dataset_name}] Building H5 from {len(files)} CSV files -> {h5_out}')
    os.makedirs(os.path.dirname(os.path.abspath(h5_out)), exist_ok=True)

    seq_counter = {} # (subj, mod, ftype, sfont, cond) -> next seq index
    n_written   = 0

    with h5py.File(h5_out, 'w') as hf:
        for fi, fname in enumerate(sorted(files)):
            fpath = os.path.join(csv_dir, fname)
            try:
                subj, mod, ftype, sfont, cond = parse_fn(fname)
            except Exception as e:
                print(f'  SKIP {fname}: parse error {e}')
                continue

            if None in (subj, mod, ftype, cond):
                continue

            try:
                seq_data = load_sequences(fpath, dataset_name, n_elec=n_elec)
            except Exception as e:
                print(f'  SKIP {fname}: load error {e}')
                continue

            sk = f'S{subj}'
            for cond_idx in sorted(seq_data.keys()):
                eeg  = seq_data[cond_idx]        # (N, n_elec)
                feat = compute_seq_features(eeg) # (68, 9)

                key = (sk, mod, ftype, sfont, cond)
                seq_i = seq_counter.get(key, 0)
                seq_counter[key] = seq_i + 1

                # navigate/create H5 groups
                grp = hf.require_group(sk)
                grp = grp.require_group(mod)
                grp = grp.require_group(ftype)
                if sfont:
                    grp = grp.require_group(sfont)
                grp = grp.require_group(cond)
                ds_name = f'sequence_{seq_i}'
                grp.create_dataset(ds_name, data=feat, dtype=np.float32)
                n_written += 1

            if (fi + 1) % 50 == 0 or fi + 1 == len(files):
                print(f'  [{fi+1}/{len(files)}] sequences written so far: {n_written}')

    print(f'[{dataset_name}] Done. Total sequences: {n_written}')

# load from H5
def load_seqlevel_h5(h5_path, dataset_name):
    """Load all sequences from a seqlevel H5 into a list of records."""
    records = []
    with h5py.File(h5_path, 'r') as hf:
        for sk in hf.keys():
            sg = hf[sk]
            for mod in sg.keys():
                mg = sg[mod]
                for ftype in mg.keys():
                    fg = mg[ftype]
                    # check if next level is subfont or condition
                    first_key = list(fg.keys())[0]
                    if first_key in ('Par', 'Control'):
                        nav_items = [(None, fg)]
                    else:
                        nav_items = [(sfont, fg[sfont]) for sfont in fg.keys()]

                    for sfont, cg in nav_items:
                        for cond in cg.keys():
                            for seq_name in cg[cond].keys():
                                feat = cg[cond][seq_name][:] # (68, 9)
                                records.append(dict(
                                    dataset=dataset_name,
                                    subject=sk,
                                    modality=mod,
                                    font_type=ftype,
                                    specific_font=sfont,
                                    condition=cond,
                                    seq_name=seq_name,
                                    feat=feat.astype(np.float32),
                                ))
    print(f'Loaded {len(records)} sequences from {h5_path}')
    return records

def build_df(records):
    """Flatten records to DataFrame with summary scalars."""
    rows = []
    for r in records:
        f = r['feat'] # (68, 9)
        # canonical font label
        ftype  = r['font_type']
        sfont  = r['specific_font']
        if sfont:
            font_label = f"{ftype}/{sfont}"
        else:
            font_label = ftype

        row = {
            'dataset':    r['dataset'],
            'subject':    r['subject'],
            'modality':   r['modality'],
            'font_label': font_label,
            'condition':  r['condition'],
            'seq_name':   r['seq_name'],
            'block_id':   f"{r['subject']}|{r['modality']}|{font_label}|{r['condition']}",
            'snr_disc_all':  float(f[:, 0].mean()),
            # global means per feature
            **{f'global_{FEAT_NAMES[fi]}': float(f[:, fi].mean())
               for fi in range(N_FEATS)},
            'flat': r['feat'].ravel(), # stored as object column
        }
        rows.append(row)
    return pd.DataFrame(rows)

# plots
def cohen_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)

def plot_snr_comparison(df, out_dir):
    combos = df[['dataset', 'modality']].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(len(combos), 2, figsize=(12, 5 * len(combos)), squeeze=False)
    fig.suptitle('SNR at 3.75 Hz — Parity vs Control\n(sequence-level FFT, occipital electrodes)',
                 fontsize=13, fontweight='bold')

    for ri, (ds, mod) in enumerate(combos):
        sub = df[(df.dataset == ds) & (df.modality == mod)]
        ax_v, ax_p = axes[ri, 0], axes[ri, 1]
        title = f'{ds} — {mod}'

        # violin
        for xi, cond in enumerate(['Par', 'Control']):
            vals = sub[sub.condition == cond]['snr_disc_all'].values
            parts = ax_v.violinplot(vals, [xi], widths=0.5, showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(COND_COLORS[cond]); pc.set_alpha(0.5)
            parts['cmeans'].set_color('black')
            rng = np.random.default_rng(RANDOM)
            ax_v.scatter(np.full(len(vals), xi) + rng.uniform(-0.1, 0.1, len(vals)),
                         vals, c=COND_COLORS[cond], alpha=0.35, s=14, zorder=3)

        ax_v.set_xticks([0, 1])
        ax_v.set_xticklabels(['Parity', 'Control'])
        ax_v.set_ylabel('Mean SNR at 3.75 Hz (all elec)')
        ax_v.set_title(title)
        ax_v.axhline(1.0, color='grey', lw=0.8, ls='--', label='SNR=1 (noise floor)')
        ax_v.legend(fontsize=8)

        # paired scatter per subject
        par_s  = sub[sub.condition == 'Par'].groupby('subject')['snr_disc_all'].mean()
        ctrl_s = sub[sub.condition == 'Control'].groupby('subject')['snr_disc_all'].mean()
        com    = par_s.index.intersection(ctrl_s.index)
        if len(com) >= 2:
            pv, cv = par_s[com].values, ctrl_s[com].values
            ax_p.scatter(cv, pv, c='steelblue', alpha=0.7, s=40, zorder=4)
            lo = min(np.concatenate([pv, cv])) * 0.97
            hi = max(np.concatenate([pv, cv])) * 1.03
            ax_p.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='unity')

            if len(com) >= 3:
                t, p = ttest_rel(pv, cv)
                d    = cohen_d_paired(pv, cv)
                pct  = (pv > cv).mean() * 100
                ax_p.text(0.04, 0.96,
                          f'n={len(com)} subj\nt={t:.2f} p={p:.3f}\nd={d:.2f}  {pct:.0f}% Par>Ctrl',
                          transform=ax_p.transAxes, va='top', fontsize=8,
                          bbox=dict(fc='white', alpha=0.7, ec='grey'))

        ax_p.set_xlabel('Control — SNR@3.75 Hz (all elec)')
        ax_p.set_ylabel('Parity — SNR@3.75 Hz (all elec)')
        ax_p.set_title(title + ' per subject')
        ax_p.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, 'snr_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_per_font(df, out_dir):
    """SNR@3.75 by font condition, one panel per font per dataset x modality."""
    combos     = df[['dataset', 'modality']].drop_duplicates().values.tolist()
    font_types = sorted(df['font_label'].unique())
    n_fonts    = len(font_types)
    fig, axes  = plt.subplots(len(combos), n_fonts,
                              figsize=(5 * n_fonts, 4.5 * len(combos)),
                              squeeze=False)
    fig.suptitle('SNR@3.75 Hz by Font — Par vs Control (sequence-level)', fontsize=12)

    for ri, (ds, mod) in enumerate(combos):
        sub = df[(df.dataset == ds) & (df.modality == mod)]
        for ci, font in enumerate(font_types):
            ax   = axes[ri, ci]
            fsub = sub[sub.font_label == font]
            if fsub.empty:
                ax.set_visible(False); continue

            for xi, cond in enumerate(['Par', 'Control']):
                vals = fsub[fsub.condition == cond]['snr_disc_all'].values
                if not len(vals): continue
                parts = ax.violinplot(vals, [xi], widths=0.5, showmeans=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(COND_COLORS[cond]); pc.set_alpha(0.5)
                ax.scatter(np.full(len(vals), xi) +
                           np.random.default_rng(RANDOM).uniform(-0.1, 0.1, len(vals)),
                           vals, c=COND_COLORS[cond], alpha=0.3, s=12)

            pv  = fsub[fsub.condition == 'Par'].groupby('subject')['snr_disc_all'].mean()
            cv  = fsub[fsub.condition == 'Control'].groupby('subject')['snr_disc_all'].mean()
            com = pv.index.intersection(cv.index)
            stat_txt = ''
            if len(com) >= 3:
                t, p = ttest_rel(pv[com].values, cv[com].values)
                d    = cohen_d_paired(pv[com].values, cv[com].values)
                stat_txt = f't={t:.2f} p={p:.3f}\nd={d:.2f}'

            ax.set_title(f'{ds} {mod}\n{font}\n{stat_txt}', fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Par', 'Ctrl'], fontsize=8)
            ax.axhline(1.0, color='grey', lw=0.7, ls='--')
            if ci == 0:
                ax.set_ylabel('SNR@3.75 Hz (all elec)')

    plt.tight_layout()
    out = os.path.join(out_dir, 'per_font_snr.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_electrode_snr(records, out_dir):
    """Per-electrode SNR@3.75 profile, Par vs Control, grouped by brain region."""
    combos = {}
    for r in records:
        key = (r['dataset'], r['modality'])
        if key not in combos:
            combos[key] = {'Par': [], 'Control': []}
        combos[key][r['condition']].append(r['feat'][:, 0]) # SNR_3.75

    region_order = ['Occipital', 'Parietooccipital', 'Parietal',
                    'Centroparietal', 'Central', 'Frontocentral', 'Frontal']
    reg_idx = {r: [ELEC_NAMES.index(e) for e in region_map.get(r, []) if e in ELEC_NAMES]
               for r in region_order}
    elec_order = [i for r in region_order for i in reg_idx[r]]
    tick_labels = [ELEC_NAMES[i] for i in elec_order]

    fig, axes = plt.subplots(1, len(combos), figsize=(9 * len(combos), 5), squeeze=False)
    for col_i, ((ds, mod), data) in enumerate(combos.items()):
        ax = axes[0, col_i]
        for cond, color in [('Par', COND_COLORS['Par']), ('Control', COND_COLORS['Control'])]:
            if not data[cond]: continue
            mat  = np.stack(data[cond])[:, elec_order]
            mean = mat.mean(axis=0)
            sem  = mat.std(axis=0) / np.sqrt(len(mat))
            x    = np.arange(len(elec_order))
            ax.plot(x, mean, color=color, lw=1.5, label=cond)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18)

        # region boundary lines
        pos = 0
        for reg in region_order:
            n = len(reg_idx[reg])
            if n == 0: continue
            if pos > 0:
                ax.axvline(pos - 0.5, color='grey', lw=0.6, ls='--', alpha=0.5)
            ax.text(pos + n / 2 - 0.5, ax.get_ylim()[1],
                    reg[:4], ha='center', va='bottom', fontsize=6, color='grey')
            pos += n

        ax.set_xticks(range(len(elec_order)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
        ax.axhline(1.0, color='black', lw=0.7, ls=':', label='SNR=1')
        ax.set_ylabel('Mean SNR at 3.75 Hz')
        ax.set_title(f'{ds} — {mod}')
        ax.legend()

    fig.suptitle('Per-Electrode SNR Profile (sequence-level FFT)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(out_dir, 'electrode_snr_profile.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def plot_umap_grid(df, out_dir):
    flat_all = np.stack(df['flat'].values) # (N, 612)
    N = len(flat_all)

    # build occipital-only and ratio-only feature matrices
    occ_cols   = [ei * N_FEATS + fi for ei in OCC_IDX for fi in range(N_FEATS)]
    ratio_cols = [ei * N_FEATS + 10 for ei in range(N_ELEC)] # SNR_ratio per electrode

    feat_sets = [
        ('All electrodes\n(612 feats)',       flat_all),
        ('Occipital only\n(81 feats)',         flat_all[:, occ_cols]),
        ('SNR ratio only\n(68 feats)',         flat_all[:, ratio_cols]),
    ]

    colorings = [
        ('Condition',  'condition',  COND_COLORS),
        ('Subject',    'subject',    None),
        ('Dataset',    'dataset',    None),
        ('Font',       'font_label', None),
    ]

    n_rows = len(feat_sets)
    n_cols = len(colorings)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
    fig.suptitle('UMAP of Sequences (subject-centred) — sequence-level FFT features',
                 fontsize=13, fontweight='bold')

    embeddings = []
    for feat_name, feat in feat_sets:
        print(f'  UMAP: {feat_name.split(chr(10))[0]} ({feat.shape[1]} features)')
        # subject-center
        fc = feat.copy().astype(np.float64)
        for subj in df['subject'].unique():
            m = (df['subject'] == subj).values
            fc[m] -= fc[m].mean(axis=0)
        Xsc  = StandardScaler().fit_transform(fc)
        emb  = umap.UMAP(**UMAP_PARAMS).fit_transform(Xsc)
        embeddings.append(emb)
        del fc, Xsc; gc.collect()

    for ri, (feat_name, feat) in enumerate(feat_sets):
        emb = embeddings[ri]
        for ci, (col_title, col, cmap_dict) in enumerate(colorings):
            ax   = axes[ri, ci]
            cats = df[col].unique()
            if cmap_dict:
                cat_colors = cmap_dict
            else:
                # matplotlib >= 3.9 removed cm.get_cmap
                pal = matplotlib.colormaps['tab20'].resampled(max(len(cats), 1))
                cat_colors = {c: pal(i) for i, c in enumerate(cats)}

            for cat in cats:
                mask = (df[col] == cat).values
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[cat_colors.get(cat, 'grey')],
                           alpha=0.5, s=15, label=str(cat), rasterized=True)

            ax.set_title(f'{col_title}\n{feat_name}', fontsize=8)
            ax.set_xlabel('UMAP 1', fontsize=7)
            ax.set_ylabel('UMAP 2', fontsize=7)
            ax.tick_params(labelsize=6)
            if len(cats) <= 12:
                ax.legend(markerscale=2, fontsize=6, framealpha=0.5,
                          loc='best', ncol=2)

    plt.tight_layout()
    out = os.path.join(out_dir, 'umap_grid.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return embeddings   # return for clustering

def plot_clustering(df, embeddings, out_dir):
    feat_names_short = ['all', 'occ', 'ratio']
    y_true = (df['condition'] == 'Par').astype(int).values

    fig, axes = plt.subplots(2, len(embeddings), figsize=(6 * len(embeddings), 10))
    fig.suptitle('Clustering of UMAP Embeddings — Par=red, Control=blue\nGMM(k=2) top | KMeans(k=2) bottom',
                 fontsize=12, fontweight='bold')

    for col_i, (emb, fsname) in enumerate(zip(embeddings, feat_names_short)):
        for row_i, (ClsModel, model_name) in enumerate([
            (GaussianMixture(n_components=2, n_init=5, random_state=RANDOM), 'GMM k=2'),
            (KMeans(n_clusters=2, n_init=10, random_state=RANDOM), 'KMeans k=2'),
        ]):
            ax = axes[row_i, col_i]
            lbls = ClsModel.fit_predict(emb)

            # compute purity: for each cluster, majority vote among Par/Control
            purity = 0.0
            for cl in np.unique(lbls):
                mask  = lbls == cl
                votes = y_true[mask]
                purity += max(votes.sum(), mask.sum() - votes.sum())
            purity /= len(y_true)

            # colour by true condition, marker by cluster
            markers = ['o', '^']
            for ci, cond in enumerate(['Par', 'Control']):
                true_mask = (df['condition'] == cond).values
                for cli in range(2):
                    m = true_mask & (lbls == cli)
                    ax.scatter(emb[m, 0], emb[m, 1],
                               c=COND_COLORS[cond], marker=markers[cli],
                               alpha=0.5, s=18, rasterized=True)

            ax.set_title(f'{fsname} feats  {model_name}\npurity={purity:.3f}', fontsize=9)
            ax.set_xlabel('UMAP 1', fontsize=7)
            ax.set_ylabel('UMAP 2', fontsize=7)

    plt.tight_layout()
    out = os.path.join(out_dir, 'clustering.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# statistics
def compute_stats(df):
    rows = []
    for (ds, mod, font), grp in df.groupby(['dataset', 'modality', 'font_label']):
        par  = grp[grp.condition == 'Par'].groupby('subject')['snr_disc_all'].mean()
        ctrl = grp[grp.condition == 'Control'].groupby('subject')['snr_disc_all'].mean()
        com  = par.index.intersection(ctrl.index)
        if len(com) < 3: continue
        pv, cv = par[com].values, ctrl[com].values
        t, p   = ttest_rel(pv, cv)
        d      = cohen_d_paired(pv, cv)
        rows.append(dict(
            dataset=ds, modality=mod, font=font,
            n_subjects=len(com),
            mean_snr_par=round(pv.mean(), 4),
            mean_snr_ctrl=round(cv.mean(), 4),
            mean_diff=round((pv - cv).mean(), 4),
            t_stat=round(t, 4),
            p_value=round(p, 4),
            cohen_d=round(d, 4),
            pct_par_above=round((pv > cv).mean() * 100, 1),
        ))
    return pd.DataFrame(rows)

def main():
    print('=' * 70)
    print('DIGNUMWORDS-SHAPE  Sequence-Level Feature Rebuild + UMAP')
    print('=' * 70)
    print(f'FFT size: {N_FFT}  ->  bin spacing: {SR/N_FFT:.5f} Hz')
    print(f'Target bins: 3.75Hz={TARGET_BINS[0]}, 7.5Hz={TARGET_BINS[1]}, '
          f'11.25Hz={TARGET_BINS[2]}, 15Hz={TARGET_BINS[3]}')
    print()

    # build H5 files
    build_h5(ANG_CSV_DIR, ANG_H5_OUT,
             parse_fn=parse_angelique_fname,
             dataset_name='Angelique',
             n_elec=N_ELEC) # drop EOG channels 68-69

    build_h5(TAL_CSV_DIR, TAL_H5_OUT,
             parse_fn=parse_talia_fname,
             get_files_fn=_get_talia_files,
             dataset_name='Talia',
             n_elec=N_ELEC)

    # load both datasets
    print('\nLoading sequences from H5')
    rec_ang = load_seqlevel_h5(ANG_H5_OUT, 'Angelique')
    rec_tal = load_seqlevel_h5(TAL_H5_OUT, 'Talia')
    records = rec_ang + rec_tal

    df = build_df(records)
    df.drop(columns=['flat']).to_csv(os.path.join(OUT_DIR, 'sequences.csv'), index=False)
    print(f'Total sequences: {len(df)}')
    print(df.groupby(['dataset', 'modality', 'font_label', 'condition']).size()
           .to_string())

    # statistics
    print('\nStatistics (SNR@3.75 Hz, paired t-test)')
    stats_df = compute_stats(df)
    stats_df.to_csv(os.path.join(OUT_DIR, 'statistics.csv'), index=False)
    for _, row in stats_df.iterrows():
        sig = ('***' if row.p_value < 0.001 else
               '**'  if row.p_value < 0.01  else
               '*'   if row.p_value < 0.05  else 'ns')
        print(f"  {row.dataset:10} {row.modality:8} {row.font:10} | "
              f"Par={row.mean_snr_par:.3f} Ctrl={row.mean_snr_ctrl:.3f} "
              f"delta={row.mean_diff:+.3f} | "
              f"t={row.t_stat:+.2f} p={row.p_value:.3f} {sig} "
              f"d={row.cohen_d:+.2f} | {row.pct_par_above:.0f}% Par>Ctrl")

    # figures
    print('\nGenerating figures')
    plot_snr_comparison(df, OUT_DIR)
    plot_per_font(df, OUT_DIR)
    plot_electrode_snr(records, OUT_DIR)
    embeddings = plot_umap_grid(df, OUT_DIR)
    plot_clustering(df, embeddings, OUT_DIR)

    print('\nDone')
    print(f'Results in: {OUT_DIR}/')
    for fname in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fname))
        print(f'  {fname:45s}  {sz/1024:8.1f} KB')

if __name__ == '__main__':
    main()
