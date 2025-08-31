import os
import h5py
import pandas as pd
import numpy as np
from electrodes import biosemi_68_order

def load_modality_data(base_dir, subjects, modality, font=None, condition=None):
    """
    Load EEG data from HDF5, filtered by modality, font, and condition.
    Supports nested fonts like '20F/A'.
    Labels are assigned as odd=1, even=0.
    """
    X, y, subj_ids = [], [], []

    h5_path = os.path.join(base_dir, 'features_per_bin.h5')
    if not os.path.exists(h5_path):
        return pd.DataFrame(), np.array([]), np.array([])

    # Split font into main font and optional subfont
    main_font, sub_font = (font.split('/') + [None])[:2] if font else (None, None)

    with h5py.File(h5_path, 'r') as f:
        for subject in subjects:
            subject_group = f"S{subject:02d}.csv"
            if subject_group not in f:
                continue
            subj_grp = f[subject_group]

            # --- filter modality ---
            if modality not in subj_grp:
                continue
            cat_grp = subj_grp[modality]

            # --- filter main font ---
            for font_type in cat_grp.keys():
                if main_font is not None and font_type != main_font:
                    continue
                font_grp = cat_grp[font_type]

                # --- filter subfont ---
                if sub_font is not None:
                    if sub_font not in font_grp:
                        continue
                    sf_grp = font_grp[sub_font]
                else:
                    sf_grp = font_grp

                # --- filter condition ---
                for cond_group_name in ['Par', 'Control']:
                    if condition is not None and cond_group_name != condition:
                        continue
                    if cond_group_name not in sf_grp:
                        continue
                    cond_grp = sf_grp[cond_group_name]

                    # --- loop sequences ---
                    for seq_name in cond_grp.keys():
                        seq_idx = int(seq_name.replace('sequence_', ''))
                        seq_grp = cond_grp[seq_name]

                        # --- loop bins ---
                        for bin_name in seq_grp.keys():
                            bin_idx = int(bin_name.split('_')[0].replace('bin',''))
                            dset = seq_grp[bin_name]
                            data = dset[:]

                            # label odd=1, even=0
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
                            df['category'] = modality
                            df['font_type'] = font_type
                            if sub_font:
                                df['specific_font'] = sub_font

                            X.append(df)
                            y.extend([label_val] * n_electrodes)
                            subj_ids.extend([subject] * n_electrodes)

    if not X:
        return pd.DataFrame(), np.array([]), np.array([])

    X_combined = pd.concat(X).reset_index(drop=True)
    y_combined = np.array(y)
    subj_ids_combined = np.array(subj_ids)

    return X_combined, y_combined, subj_ids_combined