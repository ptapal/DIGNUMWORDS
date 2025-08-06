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
    """Electrode analysis with region importance and mixed evaluation support"""
    new_dir = os.path.join(output_dir, f'{mode}_{train_mod}_{test_mod}_{condition_train}_{condition_test}_{font_train}_{font_test}')
    os.makedirs(new_dir, exist_ok=True)
    if condition_test is None:
        condition_test = condition_train
    if font_test is None:
        font_test = font_train

    # Handle different evaluation modes
    if mode == 'mixed':
        # Load both Dig and NumWo for training
        X_train_dig, y_train_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font_train, condition_train)
        X_train_num, y_train_num, _ = load_modality_data(base_dir, subjects, 'NumWo', font_train, condition_train)
        X_train = pd.concat([X_train_dig, X_train_num])
        y_train = pd.Series(np.concatenate([y_train_dig, y_train_num]), index=X_train.index)
        X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)
    else:
        # Standard within or cross evaluation
        X_train, y_train, _ = load_modality_data(base_dir, subjects, train_mod, font_train, condition_train)
        if mode == 'within' and train_mod == test_mod and font_train == font_test and condition_train == condition_test:
            X_test, y_test = X_train.copy(), y_train.copy()
        else:
            X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)

    # Get electrode positions
    positions = get_biosemi68_positions()
    ch_names = list(positions.keys())
    pos = np.array([positions[ch] for ch in ch_names])[:, :2]
    
    # Initialize importance storage
    electrode_importance = {}
    region_scores = {region: [] for region in region_map.keys()}
    
    # Calculate electrode importance
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
            
            # Store electrode importance
            electrode_importance[electrode] = score
            
            # Aggregate to regions
            for region, electrodes in region_map.items():
                if electrode in electrodes:
                    region_scores[region].append(score)
    
    # Calculate mean region importance
    region_importance = {k: np.mean(v) if v else 0 for k, v in region_scores.items()}
    
    # Create optimized topomap
    plt.figure(figsize=(12, 6))
    
    # Make the actual topomap larger
    ax = plt.subplot(121, aspect='equal')  # Square aspect ratio
    
    data = np.array([electrode_importance.get(ch, 0) for ch in ch_names])
    
    im, _ = plot_topomap(
        data,
        pos,
        names=ch_names,
        cmap='RdBu_r',
        show=False,
        axes=ax,
        sensors=True,
        res=128,
        outlines='head',
        extrapolate='local',
        size=2  # Larger electrode markers
    )
    
    # Add colorbar with reasonable size
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Balanced Accuracy', size=10)
    
    # Add table with top electrodes on the right
    top_electrodes = pd.Series(electrode_importance).sort_values(ascending=False).head(10)
    ax2 = plt.subplot(122)
    ax2.axis('off')
    table = ax2.table(
        cellText=np.round(top_electrodes.values.reshape(-1, 1), 3),
        rowLabels=top_electrodes.index,
        colLabels=['Score'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    
    plt.suptitle(
        f'Electrode Importance ({strategy} strategy)\n{mode} {train_mod}→{test_mod}, {condition_train}→{condition_test}, {font_train}→{font_test}',
        y=1.0,
        fontsize=7
    )
    plt.tight_layout()

    plot_path = os.path.join(new_dir, 'electrode_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    electrode_df = pd.DataFrame.from_dict(electrode_importance, orient='index', columns=['balanced_accuracy'])
    region_df = pd.DataFrame.from_dict(region_importance, orient='index', columns=['balanced_accuracy'])
    
    electrode_df.to_csv(os.path.join(new_dir, 'electrode_results.csv'))
    region_df.to_csv(os.path.join(new_dir, 'region_results.csv'))
    
    return electrode_df, region_df

base_dir = 'h5_sep'
subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]

'''results = analyze_electrodes(
    base_dir,
    subjects,
    train_mod='Dig',
    test_mod='Dig',
    condition_train='Parity',
    condition_test='Parity',
    font_train='1F',
    font_test='1F',
    mode='within'
)'''

# Configuration for all possible runs
modalities = ['Dig', 'NumWo']
modes = ['within', 'cross', 'mixed']
fonts = ['1F', '20F/A', '20F/S']
conditions = ['Parity', 'Control']
subjects = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28]


for mode in modes:
    for train_mod in modalities:
        for test_mod in modalities:
            # Skip invalid combinations
            if mode == 'within' and train_mod != test_mod:
                continue
            if mode == 'mixed' and train_mod != 'Dig':  # Mixed always combines Dig+Num
                continue
            
            # Font transitions (same or different)
            for font_train in fonts:
                for font_test in fonts:
                    # Condition transitions
                    for condition_train in conditions:
                        for condition_test in conditions:
                            # Apply transition rules
                            if font_train != font_test and condition_train != condition_test:
                                continue  # No simultaneous font+condition transition
                            
                            # Handle mixed mode specially
                            if mode == 'mixed':
                                print(f"\nRunning MIXED evaluation (Dig+NumWo→{test_mod})")
                                print(f"Font: {font_train}→{font_test}")
                                print(f"Condition: {condition_train}→{condition_test}")
                                
                                results = analyze_electrodes(
                                    base_dir='h5_sep',
                                    subjects=subjects,
                                    train_mod='Dig',  # Will be combined with NumWo internally
                                    test_mod=test_mod,
                                    condition_train=condition_train,
                                    condition_test=condition_test,
                                    font_train=font_train,
                                    font_test=font_test,
                                    mode='mixed',
                                    output_dir='results/mixed'
                                )
                            else:
                                # Standard within/cross evaluation
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