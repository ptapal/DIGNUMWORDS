from operator import pos
import os
import h5py
import pandas as pd
import numpy as np
from electrodes import get_biosemi68_positions, region_map
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from mne.viz import plot_topomap
from matplotlib import pyplot as plt
from load_data import load_modality_data
import mne

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

    if mode == 'mixed':
        X_train_dig, y_train_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font_train, condition_train)
        X_train_num, y_train_num, _ = load_modality_data(base_dir, subjects, 'NumWo', font_train, condition_train)
        X_train = pd.concat([X_train_dig, X_train_num])
        y_train = pd.Series(np.concatenate([y_train_dig, y_train_num]), index=X_train.index)
        X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)
    else:
        X_train, y_train, _ = load_modality_data(base_dir, subjects, train_mod, font_train, condition_train)
        if mode == 'within' and train_mod == test_mod and font_train == font_test and condition_train == condition_test:
            X_test, y_test = X_train.copy(), y_train.copy()
        else:
            X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)

    positions = get_biosemi68_positions()
    ch_names = list(positions.keys())
    pos = np.array([positions[ch] for ch in ch_names])[:, :2]
    
    electrode_importance = {}
    region_scores = {region: [] for region in region_map.keys()}
    
    for electrode in X_train['Electrode'].unique():
        if strategy == 'drop':
            train_mask = X_train['Electrode'] != electrode
            test_mask = X_test['Electrode'] != electrode
        else:
            train_mask = X_train['Electrode'] == electrode
            test_mask = X_test['Electrode'] == electrode
            
        X_train_filt = X_train[train_mask].drop(columns=['Electrode'])
        X_test_filt = X_test[test_mask].drop(columns=['Electrode'])
        
        if len(X_train_filt) > 0:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(random_state=random_state))
            ])
            model.fit(X_train_filt, y_train[train_mask])
            y_pred = model.predict(X_test_filt)
            score = balanced_accuracy_score(y_test[test_mask], y_pred)
            
            electrode_importance[electrode] = score

            for region, electrodes in region_map.items():
                if electrode in electrodes:
                    region_scores[region].append(score)
    
    region_importance = {k: np.mean(v) if v else 0 for k, v in region_scores.items()}


    plt.figure(figsize=(8,8))

    '''
    # Filter out EOG and external electrodes for visualization
    keep_mask = [not (ch.startswith('E') or ch.startswith('EOG')) for ch in ch_names]
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    data = np.array([electrode_importance[ch_names[i]] for i in keep_indices])
    pos_filtered = np.array([pos[i] for i in keep_indices])
    '''
    data = np.array([electrode_importance[ch] for ch in ch_names])
    pos_filtered = pos  
    assert len(data) == len(pos_filtered), "Data and positions must have same length"

    ax = plt.subplot(111, aspect='equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    data_mean = np.mean(data)
    if data_mean < 0.6:
        center = 0.5
        data_range = max(abs(data - center))
        vmin = max(0, center - data_range * 1.5)
        vmax = min(1, center + data_range * 1.5)
        cmap = 'RdBu_r'
    else:
        range_padding = (max(data) - min(data)) * 0.3
        vmin = max(0.5, min(data) - range_padding)
        vmax = min(1.0, max(data) + range_padding)
        cmap = 'viridis' if data_mean > 0.7 else 'plasma'

    im = plot_topomap(
    data,
    pos_filtered,
    cmap='RdBu_r',
    axes=ax,
    show=False,
    sensors=False,
    res=300,
    outlines='head',
    extrapolate='head', 
    border=0,
    vlim=(vmin, vmax) 
    )[0]

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off') 

    cax = ax.inset_axes([0.78, 0.08, 0.03, 0.2])  
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Balanced Accuracy', size=12)

    plt.savefig(
        os.path.join(new_dir, 'electrode_importance.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0 
    )
    plt.close()
    
    electrode_df = pd.DataFrame.from_dict(electrode_importance, orient='index', columns=['balanced_accuracy'])
    region_df = pd.DataFrame.from_dict(region_importance, orient='index', columns=['balanced_accuracy'])
    
    electrode_df.to_csv(os.path.join(new_dir, 'electrode_results.csv'))
    region_df.to_csv(os.path.join(new_dir, 'region_results.csv'))
    
    return electrode_df, region_df

base_dir = 'h5_sep'
subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]

modalities = ['Dig', 'NumWo']
modes = ['within', 'cross', 'mixed']
fonts = ['1F', '20F/A', '20F/S']
conditions = ['Parity', 'Control']


for mode in modes:
    for train_mod in modalities:
        for test_mod in modalities:
           
            if mode == 'within' and train_mod != test_mod:
                continue
            if mode == 'mixed' and train_mod != 'Dig':  
                continue
            
            for font_train in fonts:
                for font_test in fonts:
                    for condition_train in conditions:
                        for condition_test in conditions:
                            if font_train != font_test and condition_train != condition_test:
                                continue 
                            
                            if mode == 'mixed':
                                print(f"\nRunning MIXED evaluation (Dig+NumWo→{test_mod})")
                                print(f"Font: {font_train}→{font_test}")
                                print(f"Condition: {condition_train}→{condition_test}")
                                
                                results = analyze_electrodes(
                                    base_dir='h5_sep',
                                    subjects=subjects,
                                    train_mod='Dig', 
                                    test_mod=test_mod,
                                    condition_train=condition_train,
                                    condition_test=condition_test,
                                    font_train=font_train,
                                    font_test=font_test,
                                    mode='mixed',
                                    output_dir='results/mixed'
                                )
                            else:
                                print(f"\nRunning {mode.upper()} evaluation ({train_mod}→{test_mod})")
                                print(f"Font: {font_train}→{font_test}")
                                print(f"Condition: {condition_train}→{condition_test}")
                                
                                results = analyze_electrodes(
                                    base_dir='h5_sep',
                                    subjects=subjects,
                                    train_mod=train_mod,
                                    test_mod=test_mod,
                                    condition_train=condition_train,
                                    condition_test=condition_test,
                                    font_train=font_train,
                                    font_test=font_test,
                                    mode=mode,
                                    output_dir=f'results/{mode}'
                                )