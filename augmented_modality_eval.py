from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, 
                            matthews_corrcoef, roc_auc_score, precision_recall_curve,
                            average_precision_score, roc_curve, log_loss, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from load_data import load_modality_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, 
                             confusion_matrix, matthews_corrcoef)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_modality_data

def aug_evaluate_modality(base_dir, subjects, modality, 
                          font_train, font_test=None, 
                          condition_train=None, condition_test=None,
                          mode='within', test_mod=None,
                          plot_time_resolved=True, test_size=0.2,
                          random_state=42):

    font_test = font_test or font_train
    condition_test = condition_test or condition_train
    condition_train = condition_train or condition_test

    if mode == 'cross':
        if not test_mod:
            raise ValueError("test_mod must be specified for cross-modality evaluation")
        X_train, y_train, _ = load_modality_data(base_dir, subjects, modality, font_train, condition_train)
        X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)
    elif mode == 'within':
        subjects = np.array(subjects)
        np.random.seed(random_state)
        np.random.shuffle(subjects)
        n_test = max(1, int(len(subjects) * test_size))
        test_subjects = subjects[:n_test]
        train_subjects = subjects[n_test:]
        X_train, y_train, _ = load_modality_data(base_dir, train_subjects, modality, font_train, condition_train)
        X_test, y_test, _ = load_modality_data(base_dir, test_subjects, modality, font_train, condition_train)
        
        if X_train.empty or X_test.empty:
            print("No data found for the selected subjects.")
            return None
    elif mode == 'mixed':
        if not test_mod:
            raise ValueError("test_mod must be specified for mixed-modality evaluation")
        X_dig, y_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font_train, condition_train)
        X_num, y_num, _ = load_modality_data(base_dir, subjects, 'NumWo', font_train, condition_train)

        X_all = pd.concat([X_dig, X_num], axis=0)
        y_all = np.concatenate([y_dig, y_num])

        subj_all = np.unique(X_all['subject'])
        np.random.seed(random_state)
        np.random.shuffle(subjects)
        n_test = max(1, int(len(subj_all) * test_size))
        test_subjects = subj_all[:n_test]
        train_subjects = subj_all[n_test:]

        train_mask = X_all['subject'].isin(train_subjects)
        X_train, y_train = X_all.loc[train_mask], y_all[train_mask]
        X_test, y_test, _ = load_modality_data(base_dir, test_subjects, test_mod, font, condition)
        if X_test.empty:
            print("No test data found for mixed modality.")
            return None
        
    else:
        raise ValueError("Invalid mode. Choose 'cross', 'within', or 'mixed'")

    if X_train.empty or X_test.empty:
        return None

    metadata_cols = ['bin', 'sequence', 'subject']
    feature_cols = X_train.select_dtypes(include=np.number).columns.difference(metadata_cols)
    X_train_num = X_train[feature_cols].copy()
    X_test_num = X_test[feature_cols].copy()
    X_train_num.columns = X_train_num.columns.astype(str)
    X_test_num.columns = X_test_num.columns.astype(str)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=random_state, n_jobs=-1))
    ])
    model.fit(X_train_num, y_train)

    y_pred = model.predict(X_test_num)
    y_proba = model.predict_proba(X_test_num)[:, 1]

    metrics = {
        'accuracy': model.score(X_test_num, y_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'train_modality': modality,
        'test_modality': test_mod if mode in ['cross', 'mixed'] else modality,
        'train_font': font_train,
        'test_font': font_test,
        'train_condition': condition_train,
        'test_condition': condition_test
    }

    if plot_time_resolved and 'bin' in X_test.columns:
        sequences = np.sort(X_test['sequence'].unique())  # get all sequences
        plt.figure(figsize=(12, 4))

        colors = ['blue', 'red']  # sequence 1 -> blue, sequence 2 -> red

        bin_acc_all = []

        for seq_idx, seq_val in enumerate(sequences):
            seq_mask = X_test['sequence'] == seq_val
            bins = np.sort(X_test.loc[seq_mask, 'bin'].unique())
            bin_acc = []

            for b in bins:
                idx = seq_mask & (X_test['bin'] == b)
                if idx.sum() == 0:
                    continue
                X_bin = X_test_num.loc[idx]
                y_bin = y_test[idx]
                y_bin_pred = model.predict(X_bin)
                acc = (y_bin_pred == y_bin).mean()
                bin_acc.append((b, acc))

            bin_acc = np.array(bin_acc)
            bin_acc_all.append(bin_acc)
            plt.plot(bin_acc[:, 0], bin_acc[:, 1], marker='o', color=colors[seq_idx % len(colors)], label=f'Sequence {seq_val}')

        plt.xlabel('Bin')
        plt.ylabel('Accuracy')
        plt.title(f'Time-resolved accuracy ({mode})')
        plt.grid(True)
        plt.legend()
        plt.show()

        metrics['time_resolved'] = bin_acc_all

    clf = model.named_steps['clf']
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    metrics['feature_importances'] = feature_importance_df

    return metrics