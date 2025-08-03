from load_data import load_modality_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef
import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import os
from mne.viz import plot_topomap
from electrodes import region_map, get_biosemi68_positions

font = '1F'  
condition = 'Par' 

def evaluate_region_dropout(
    base_dir, subjects, train_mod, test_mod, electrode_groups, 
    max_comb_size=2, strategy="drop", metric="balanced_accuracy",
    output_dir=os.path.join("results", "region_importances", f"{font}_{condition}")
):
    os.makedirs(output_dir, exist_ok=True)
    
    electrode_positions = get_biosemi68_positions()
    ch_names = list(electrode_positions.keys())
    pos = np.array([electrode_positions[ch] for ch in ch_names])[:, :2]
    
    full_results = {}
    region_scores = {region: [] for region in electrode_groups.keys()}
    electrode_importance = {ch: [] for ch in ch_names if ch not in ['E1','E2','E3','E4','E5','E6','EOG1','EOG2']}

    X_train, y_train, _ = load_modality_data(base_dir, subjects, train_mod, font, condition)
    X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font, condition)

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train.flatten(), index=X_train.index)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test.flatten(), index=X_test.index)

    if 'Electrode' not in X_train.columns:
        raise ValueError("Electrode column not found in dataset.")

    for r in range(1, max_comb_size + 1):
        for region_comb in combinations(electrode_groups.keys(), r):
            region_label = "+".join(region_comb)

            if strategy == "drop":
                channels_to_remove = sum([electrode_groups[region] for region in region_comb], [])
                keep_train = ~X_train['Electrode'].isin(channels_to_remove)
                keep_test = ~X_test['Electrode'].isin(channels_to_remove)
            elif strategy == "keep":
                channels_to_keep = sum([electrode_groups[region] for region in region_comb], [])
                keep_train = X_train['Electrode'].isin(channels_to_keep)
                keep_test = X_test['Electrode'].isin(channels_to_keep)
            else:
                raise ValueError("strategy must be 'drop' or 'keep'")

            X_train_filt = X_train[keep_train].drop(columns=['Electrode'])
            y_train_filt = y_train[keep_train]
            X_test_filt = X_test[keep_test].drop(columns=['Electrode'])
            y_test_filt = y_test[keep_test]

            if len(X_train_filt) == 0:
                print(f"Skipping {region_label} — no training data left.")
                continue

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(eval_metric='logloss'))
            ])
            
            try:
                pipe.fit(X_train_filt, y_train_filt)
                y_pred = pipe.predict(X_test_filt)
                y_proba = pipe.predict_proba(X_test_filt)[:, 1] if hasattr(pipe, 'predict_proba') else None
                
                if metric == "balanced_accuracy":
                    score = balanced_accuracy_score(y_test_filt, y_pred)
                elif metric == "roc_auc" and y_proba is not None:
                    score = roc_auc_score(y_test_filt, y_proba)
                elif metric == "mcc":
                    score = matthews_corrcoef(y_test_filt, y_pred)
                else:
                    raise ValueError("Unsupported metric or missing predict_proba")

                full_results[region_label] = score
                
                for region in region_comb:
                    region_scores[region].append(score)
                    for electrode in electrode_groups[region]:
                        if electrode in electrode_importance:
                            electrode_importance[electrode].append(score)
                
            except Exception as e:
                print(f"Error with region {region_label}: {e}")
                continue

    results_df = pd.DataFrame.from_dict(full_results, orient='index', columns=[metric])
    results_df.to_csv(os.path.join(output_dir, f"{font}_{condition}_region_results_{strategy}.csv"))

    region_importance = {region: np.mean(scores) for region, scores in region_scores.items()}
    electrode_importance = {ch: np.mean(scores) if scores else 0 for ch, scores in electrode_importance.items()}
    
    region_df = pd.DataFrame.from_dict(region_importance, orient='index', columns=['importance'])
    electrode_df = pd.DataFrame.from_dict(electrode_importance, orient='index', columns=['importance'])
    
    region_df.to_csv(os.path.join(output_dir, f"{font}_{condition}_region_importance_{strategy}.csv"))
    electrode_df.to_csv(os.path.join(output_dir, f"{font}_{condition}_electrode_importance_{strategy}.csv"))

    fig, ax = plt.subplots(figsize=(12, 8))
    
    data = np.zeros(len(ch_names))
    for i, ch in enumerate(ch_names):
        if ch in electrode_importance:
            data[i] = electrode_importance[ch]
    
    # eeg_mask = [i for i, ch in enumerate(ch_names) if ch not in ['E1','E2','E3','E4','E5','E6','EOG1','EOG2']]
    eeg_mask = [i for i, ch in enumerate(ch_names)]
    
    im, _ = plot_topomap(
        data[eeg_mask],
        pos[eeg_mask],
        names=np.array(ch_names)[eeg_mask],
        cmap='RdBu_r',
        show=False,
        axes=ax,
        outlines='head',
        sensors=True,
        res=128
    )
    im.set_clim(np.nanmin(data[eeg_mask]), np.nanmax(data[eeg_mask]))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Importance Score ({metric})')
    ax.set_title(f'Electrode Importance ({strategy} strategy)\nBioSemi 68-Channel')
    
    plot_path = os.path.join(output_dir, f"{font}_{condition}_topomap_{strategy}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {output_dir}")
    return results_df, region_df, electrode_df


'''
#Example usage

base_dir = 'h5_sep'
subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]

results, region_imp, electrode_imp = evaluate_region_dropout(
    base_dir,
    subjects,
    train_mod="Dig",
    test_mod="Dig",
    electrode_groups=region_map,
    max_comb_size=2,
    strategy="drop",
    metric="balanced_accuracy"
)
'''