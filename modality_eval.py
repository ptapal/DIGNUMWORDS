import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from load_data import load_modality_data

def evaluate_modality(base_dir, subjects, modality, font, condition,
                      mode='within', test_mod=None, plot_time_resolved=True,
                      test_size=0.2, random_state=42):

    if mode == 'within':
        subjects = np.array(subjects)
        np.random.seed(random_state)
        np.random.shuffle(subjects)
        n_test = max(1, int(len(subjects) * test_size))
        test_subjects = subjects[:n_test]
        train_subjects = subjects[n_test:]

        X_train, y_train, _ = load_modality_data(base_dir, train_subjects, modality, font, condition)
        X_test, y_test, _ = load_modality_data(base_dir, test_subjects, modality, font, condition)

        if X_train.empty or X_test.empty:
            print("No data found for the selected subjects.")
            return None
    elif mode == 'cross':
        X_train, y_train, subj_train = load_modality_data(base_dir, subjects, modality, font, condition)
        X_test, y_test, subj_test = load_modality_data(base_dir, subjects, test_mod, font, condition)
        if X_train.empty or X_test.empty:
            print("No data found for cross modality.")
            return None
    elif mode == 'mixed':
        X_dig, y_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font, condition)
        X_num, y_num, _ = load_modality_data(base_dir, subjects, 'NumWo', font, condition)
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
        raise ValueError("Invalid mode")

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

    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.named_steps['clf'].feature_importances_
    }).sort_values(by='importance', ascending=False)

    metrics = {
        'accuracy': model.score(X_test_num, y_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'train_modality': modality,
        'test_modality': test_mod if mode in ['cross', 'mixed'] else modality,
        'feature_importances': feature_importance_df
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
    feature_names = list(feature_cols)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    metrics['feature_importances'] = feature_importance_df

    return metrics