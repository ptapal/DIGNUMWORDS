import os
import h5py
import pandas as pd
import numpy as np

from electrodes import biosemi_68_order

def load_modality_data(base_dir, subjects, modality, font=None, condition=None):
    X, y, subj_ids = [], [], []

    h5_path = os.path.join(base_dir, 'features_per_bin.h5')
    if not os.path.exists(h5_path):
        return pd.DataFrame(), np.array([]), np.array([])

    with h5py.File(h5_path, 'r') as f:
        for subject in subjects:
            subject_group = f"S{subject:02d}.csv"
            if subject_group not in f:
                continue
            subj_grp = f[subject_group]

            for category in subj_grp.keys():
                cat_grp = subj_grp[category]
                for font_type in cat_grp.keys():
                    font_grp = cat_grp[font_type]

                    font_keys = [k for k in font_grp.keys() if k not in ['Par', 'Control']]
                    if font_keys:
                        font_iter = font_keys
                    else:
                        font_iter = [None]

                    for sf in font_iter:
                        sf_grp = font_grp[sf] if sf else font_grp

                        for cond_group_name in ['Par', 'Control']:
                            if cond_group_name not in sf_grp:
                                continue
                            cond_grp = sf_grp[cond_group_name]

                            for seq_name in cond_grp.keys():
                                seq_idx = int(seq_name.replace('sequence_', ''))
                                seq_grp = cond_grp[seq_name]

                                for bin_name in seq_grp.keys():
                                    bin_idx = int(bin_name.split('_')[0].replace('bin',''))
                                    dset = seq_grp[bin_name]
                                    data = dset[:]

                                    # NEW: label odd=1, even=0 from bin_name
                                    if 'odd' in bin_name.lower():
                                        label_val = 1
                                    elif 'even' in bin_name.lower():
                                        label_val = 0
                                    else:
                                        continue

                                    n_electrodes = len(biosemi_68_order)
                                    n_features = data.size // n_electrodes
                                    df = pd.DataFrame(
                                        data.reshape(n_electrodes, n_features),
                                        index=biosemi_68_order
                                    )
                                    df['bin'] = bin_idx
                                    df['sequence'] = seq_idx
                                    df['subject'] = subject
                                    df['category'] = category
                                    df['font_type'] = font_type
                                    if sf:
                                        df['specific_font'] = sf

                                    X.append(df)
                                    y.extend([label_val] * n_electrodes)
                                    subj_ids.extend([subject] * n_electrodes)

    if not X:
        return pd.DataFrame(), np.array([]), np.array([])

    X_combined = pd.concat(X).reset_index(drop=True)
    y_combined = np.array(y)
    subj_ids_combined = np.array(subj_ids)

    return X_combined, y_combined, subj_ids_combined