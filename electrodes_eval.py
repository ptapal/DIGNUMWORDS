import os
import h5py
import pandas as pd
import numpy as np
from electrodes import biosemi_68_order, region_map, get_biosemi68_mne_montage
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from mne.viz import plot_topomap
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

def load_modality_data_new(base_dir, subjects, modality, font=None, condition=None):
    """
    Load data from the new HDF5 structure into long-format DataFrame
    with columns: electrode values + 'Electrode', 'subject', 'bin', 'sequence', 'category', 'font_type', 'specific_font'
    """
    X_list, y_list = [], []

    h5_path = os.path.join(base_dir, 'features_per_bin.h5')
    if not os.path.exists(h5_path):
        return pd.DataFrame(), np.array([])

    main_font, sub_font = (font.split('/') + [None])[:2] if font else (None, None)

    with h5py.File(h5_path, 'r') as f:
        for subj in subjects:
            subj_group = f"S{subj:02d}.csv"
            if subj_group not in f:
                continue
            subj_grp = f[subj_group]

            if modality not in subj_grp:
                continue
            mod_grp = subj_grp[modality]

            for font_type in mod_grp.keys():
                if main_font and font_type != main_font:
                    continue
                font_grp = mod_grp[font_type]

                if sub_font:
                    if sub_font not in font_grp:
                        continue
                    sf_grp = font_grp[sub_font]
                else:
                    sf_grp = font_grp

                for cond in ['Par', 'Control']:
                    if condition and cond != condition:
                        continue
                    if cond not in sf_grp:
                        continue
                    cond_grp = sf_grp[cond]

                    for seq_name in cond_grp.keys():
                        seq_idx = int(seq_name.replace('sequence_', ''))
                        seq_grp = cond_grp[seq_name]

                        for bin_name in seq_grp.keys():
                            bin_idx = int(bin_name.split('_')[0].replace('bin',''))
                            dset = seq_grp[bin_name]
                            data = dset[:]

                            # determine label
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
                            ).reset_index().melt(id_vars='index', var_name='trial', value_name='value')
                            df = df.rename(columns={'index':'Electrode'})
                            df['subject'] = subj
                            df['bin'] = bin_idx
                            df['sequence'] = seq_idx
                            df['category'] = modality
                            df['font_type'] = font_type
                            if sub_font:
                                df['specific_font'] = sub_font
                            df['label'] = label_val

                            X_list.append(df)

    if not X_list:
        return pd.DataFrame(), np.array([])

    X_combined = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_combined = X_combined['label'].values
    return X_combined, y_combined

def analyze_electrodes(
    base_dir,
    subjects,
    train_mod=None,
    test_mod=None,
    condition_train=None,
    condition_test=None,
    font_train=None,
    font_test=None,
    strategy='drop',
    mode='within',
    output_dir='results',
    random_state=42
):
    new_dir = os.path.join(output_dir, f'{mode}_{train_mod}_{test_mod}_{condition_train}_{condition_test}_{font_train}_{font_test}')
    os.makedirs(new_dir, exist_ok=True)
    if condition_test is None:
        condition_test = condition_train
    if font_test is None:
        font_test = font_train

    # --- Load training data ---
    if mode == 'mixed':
        X_train_dig, y_train_dig = load_modality_data_new(base_dir, subjects, 'Dig', font_train, condition_train)
        X_train_num, y_train_num = load_modality_data_new(base_dir, subjects, 'NumWo', font_train, condition_train)
        X_train = pd.concat([X_train_dig, X_train_num], ignore_index=True)
        y_train = np.concatenate([y_train_dig, y_train_num])
    else:
        X_train, y_train = load_modality_data_new(base_dir, subjects, train_mod, font_train, condition_train)

    if X_train.empty:
        print("No training data found!")
        return None

    # --- Load test data ---
    if mode == 'within' and train_mod == test_mod and font_train == font_test and condition_train == condition_test:
        # sample 80/20 split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )
    else:
        X_test, y_test = load_modality_data_new(base_dir, subjects, test_mod, font_test, condition_test)
        if X_test.empty:
            print("No test data found!")
            return None

    # --- Electrode-wise evaluation ---
    montage = get_biosemi68_mne_montage()
    ch_pos_dict = montage.get_positions()['ch_pos']  # dict: channel -> 3D pos
    ch_names = list(ch_pos_dict.keys())
    pos = np.array([ch_pos_dict[ch][:2] for ch in ch_names])  # take x,y only

    # Pivot all electrodes once
    X_train_pivot = (
        X_train
        .groupby(['subject','sequence','bin','trial','Electrode'])['value']
        .mean()               # or .first() if you prefer
        .unstack('Electrode')
        .fillna(0)
    )

    X_test_pivot = (
        X_test
        .groupby(['subject','sequence','bin','trial','Electrode'])['value']
        .mean()
        .unstack('Electrode')
        .fillna(0)
    )

    # Labels matching pivot index
    y_train_pivot = X_train.drop_duplicates(subset=['subject','sequence','bin','trial']).set_index(
        ['subject','sequence','bin','trial']
    )['label'].loc[X_train_pivot.index]

    y_test_pivot = X_test.drop_duplicates(subset=['subject','sequence','bin','trial']).set_index(
        ['subject','sequence','bin','trial']
    )['label'].loc[X_test_pivot.index]

    electrode_importance = {}
    region_scores = {region: [] for region in region_map.keys()}

    for electrode in X_train['Electrode'].unique():
        # Apply drop/keep strategy
        if strategy == 'drop':
            X_train_filt = X_train_pivot.drop(columns=[electrode])
            X_test_filt  = X_test_pivot.drop(columns=[electrode])
        else:
            X_train_filt = X_train_pivot[[electrode]]
            X_test_filt  = X_test_pivot[[electrode]]

        if len(X_train_filt) == 0 or len(X_test_filt) == 0:
            continue

        # Fit model with correctly aligned labels
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(random_state=random_state))
        ])
        model.fit(X_train_filt, y_train_pivot)
        y_pred = model.predict(X_test_filt)
        score = balanced_accuracy_score(y_test_pivot, y_pred)
        electrode_importance[electrode] = score

        for region, electrodes in region_map.items():
            if electrode in electrodes:
                region_scores[region].append(score)

    region_importance = {k: np.mean(v) if v else 0 for k, v in region_scores.items()}

    # --- Topomap ---
    ## Only keep electrodes that exist in your dataset
    ch_names = [ch for ch in ch_names if ch in electrode_importance]
    pos = np.array([ch_pos_dict[ch][:2] for ch in ch_names])
    data = np.array([electrode_importance[ch] for ch in ch_names])
    pos_filtered = pos  # keep same name for plot_topomap

    fig, ax = plt.subplots(figsize=(8,8))
    # This gives more space for the head
    ax_head = fig.add_axes([0.25, 0.25, 0.4, 0.6]) # left, bottom, width, height

    # Compute center around 0.5 for subtle differences
    center = 0.5
    data_range = max(abs(data - center))
    vmin = max(0, center - data_range * 1.5)  # exaggerate differences
    vmax = min(1, center + data_range * 1.5)

    # Optional: choose colormap based on mean
    cmap = 'RdBu_r' if np.mean(data) < 0.55 else 'viridis'

    im = plot_topomap(
        data,
        pos_filtered,
        cmap=cmap,
        axes=ax_head,
        show=False,
        sensors=True,
        res=300,
        outlines='head',
        extrapolate='head',
        border=0,
        vlim=(vmin, vmax)
    )[0]

    ax_head.axis('off')

    cax = ax.inset_axes([0.88, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Balanced Accuracy', size=12)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(new_dir, 'electrode_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save CSV ---
    electrode_df = pd.DataFrame.from_dict(electrode_importance, orient='index', columns=['balanced_accuracy'])
    region_df = pd.DataFrame.from_dict(region_importance, orient='index', columns=['balanced_accuracy'])
    electrode_df.to_csv(os.path.join(new_dir, 'electrode_results.csv'))
    region_df.to_csv(os.path.join(new_dir, 'region_results.csv'))

    return electrode_df, region_df


subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]

modalities = ['Dig', 'NumWo']
modes = ['mixed', 'within', 'cross']
fonts = ['1F', '20F/A', '20F/S']
condition = 'Par'


for mode in modes:
    for train_mod in modalities:
        for test_mod in modalities:

            # --- FILTERING ---
            if mode == 'within' and train_mod != test_mod:
                continue
            if mode == 'cross' and train_mod == test_mod:
                continue
            if mode == 'mixed' and train_mod != 'Dig':
                continue
            
            for font_train in fonts:
                for font_test in fonts:
                            if mode == 'mixed':
                                print(f"\nRunning MIXED evaluation (Dig+NumWo→{test_mod})")
                                print(f"Font: {font_train}→{font_test}")
                                print(f"Condition: {condition}")
                                
                                results = analyze_electrodes(
                                    base_dir='h5_freq',
                                    subjects=subjects,
                                    train_mod='Dig', 
                                    test_mod=test_mod,
                                    condition_train=condition,
                                    condition_test=condition,
                                    font_train=font_train,
                                    font_test=font_test,
                                    mode='mixed',
                                    output_dir='results/mixed'
                                )
                            else:
                                print(f"\nRunning {mode.upper()} evaluation ({train_mod}→{test_mod})")
                                print(f"Font: {font_train}→{font_test}")
                                print(f"Condition: {condition}")
                                
                                results = analyze_electrodes(
                                    base_dir='h5_freq',
                                    subjects=subjects,
                                    train_mod=train_mod,
                                    test_mod=test_mod,
                                    condition_train=condition,
                                    condition_test=condition,
                                    font_train=font_train,
                                    font_test=font_test,
                                    mode=mode,
                                    output_dir=f'results/{mode}'
                                )