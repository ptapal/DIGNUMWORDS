from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, 
                            matthews_corrcoef, roc_auc_score, precision_recall_curve,
                            average_precision_score, roc_curve, log_loss, brier_score_loss)
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
from load_data import load_modality_data

def aug_evaluate_modality(base_dir, subjects, modality, 
                     font_train, font_test=None, 
                     condition_train=None, condition_test=None,
                     mode='within', test_mod=None):
    """
    Unified evaluation function for cross-font/condition/modality analysis
    
    Parameters:
    - mode: 'cross', 'within', or 'mixed'
    - test_mod: Required for 'cross'/'mixed' modes (test modality)
    - font_train/font_test: Specify different fonts for train/test
    - condition_train/condition_test: Specify different conditions for train/test
    """
    
    font_test = font_test if font_test is not None else font_train
    condition_test = condition_test if condition_test is not None else condition_train
    condition_train = condition_train if condition_train is not None else condition_test  

    if mode == 'cross':
        if not test_mod:
            raise ValueError("test_mod must be specified for cross-modality evaluation")
        X_train, y_train, _ = load_modality_data(base_dir, subjects, modality, font_train, condition_train)
        X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)
    elif mode == 'within':
        X_train, y_train, _ = load_modality_data(base_dir, subjects, modality, font_train, condition_train)
        if font_train == font_test and condition_train == condition_test:
            X_test, y_test = X_train.copy(), y_train.copy()
        else:
            X_test, y_test, _ = load_modality_data(base_dir, subjects, modality, font_test, condition_test)
    elif mode == 'mixed':
        if not test_mod:
            raise ValueError("test_mod must be specified for mixed-modality evaluation")
        X_train_dig, y_train_dig, _ = load_modality_data(base_dir, subjects, 'Dig', font_train, condition_train)
        X_train_words, y_train_words, _ = load_modality_data(base_dir, subjects, 'NumWo', font_train, condition_train)
        X_train = pd.concat([X_train_dig, X_train_words], axis=0)
        y_train = pd.Series(np.concatenate([y_train_dig, y_train_words]), name='label')

        X_test, y_test, _ = load_modality_data(base_dir, subjects, test_mod, font_test, condition_test)
    else:
        raise ValueError("Invalid mode. Choose 'cross', 'within', or 'mixed'")

    if X_train.empty or X_test.empty:
        return None

    for df in [X_train, X_test]:
        if 'Electrode' in df.columns:
            df.drop(columns=['Electrode'], inplace=True)

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
        'support': len(y_test),
        'mode': mode,
        'train_modality': modality,
        'test_modality': test_mod if mode in ['cross', 'mixed'] else modality,
        'train_font': font_train,
        'test_font': font_test,
        'train_condition': condition_train,
        'test_condition': condition_test
    }
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    metrics.update({
        'precision_recall': (precision, recall),
        'avg_precision': average_precision_score(y_test, y_proba),
        'roc_curve': (fpr, tpr),
        'log_loss': log_loss(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'calibration': calibration_curve(y_test, y_proba, n_bins=10)
    })

    clf = model.named_steps['clf']
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    metrics['feature_importances'] = feature_importance_df
    
    return metrics