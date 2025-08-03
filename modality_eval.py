from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from load_data import load_modality_data
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                                roc_curve, log_loss, brier_score_loss,
                                balanced_accuracy_score, matthews_corrcoef,
                                roc_auc_score)
from sklearn.calibration import calibration_curve
import numpy as np

def evaluate_cross_modality(base_dir, subjects, train_mod, test_mod, font, condition):
    X_train, y_train, _ = load_modality_data(base_dir, subjects, train_mod, font, condition)
    X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font, condition)

    if X_train.empty or X_test.empty:
        return None
    
    if 'Electrode' in X_train.columns:
        X_train = X_train.drop(columns=['Electrode'])
    if 'Electrode' in X_test.columns:
        X_test = X_test.drop(columns=['Electrode'])

    feature_names = X_train.columns

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'support': len(y_test)
    }
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    loss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    metrics.update({
        'precision_recall': (precision, recall),
        'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr),
        'log_loss': loss,
        'brier_score': brier,
        'calibration': calibration_curve(y_test, y_proba, n_bins=10)
    })

    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    metrics['feature_importances'] = feature_importance_df
    
    return metrics

def evaluate_within_modality(base_dir, subjects, modality, font, condition):
    X, y, _ = load_modality_data(base_dir, subjects, modality, font, condition)

    if X.empty:
        return None
    
    if 'Electrode' in X.columns:
        X = X.drop(columns=['Electrode'])

    feature_names = X.columns

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=42))
    ])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': model.score(X, y),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'mcc': matthews_corrcoef(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'support': len(y)
    }
    
    precision, recall, _ = precision_recall_curve(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)
    loss = log_loss(y, y_proba)
    brier = brier_score_loss(y, y_proba)
    
    metrics.update({
        'precision_recall': (precision, recall),
        'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr),
        'log_loss': loss,
        'brier_score': brier,
        'calibration': calibration_curve(y, y_proba, n_bins=10)
    })

    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    metrics['feature_importances'] = feature_importance_df
    
    return metrics

def evaluate_mixed_modality(base_dir, subjects, test_mod, font, condition):
    X_train_dig, y_train_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font, condition)
    X_train_words, y_train_words, _ = load_modality_data(base_dir, subjects, 'NumWo', font, condition)

    X_train = pd.concat([X_train_dig, X_train_words], axis=0)
    y_train = pd.Series(np.concatenate([y_train_dig, y_train_words]), name='label')

    X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font, condition)

    if 'Electrode' in X_train.columns:
        X_train = X_train.drop(columns=['Electrode'])
    if 'Electrode' in X_test.columns:
        X_test = X_test.drop(columns=['Electrode'])

    feature_names = X_train.columns

    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'support': len(y_test)
    }
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    loss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    metrics.update({
        'precision_recall': (precision, recall),
        'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr),
        'log_loss': loss,
        'brier_score': brier,
        'calibration': calibration_curve(y_test, y_proba, n_bins=10)
    })

    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    metrics['feature_importances'] = feature_importance_df

    return metrics