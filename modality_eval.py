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

def evaluate_cross_modality(base_dir, subjects, train_mod, test_mod):
    X_train, y_train, _ = load_modality_data(base_dir, subjects, train_mod)
    X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod)
    
    if X_train.empty or X_test.empty:
        return None
    
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
    
    return metrics

def evaluate_within_modality(base_dir, subjects, modality):
    X, y, _ = load_modality_data(base_dir, subjects, modality)
    
    if X.empty:
        return None
    
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
    
    return metrics