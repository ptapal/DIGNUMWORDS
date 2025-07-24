from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_diagnostics(metrics, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].bar(['Accuracy', 'Balanced Acc', 'ROC AUC', 'MCC'],
             [metrics['accuracy'], metrics['balanced_accuracy'],
             metrics['roc_auc'], metrics['mcc']])
    ax[0].set_ylim(0, 1)
    ax[0].axhline(0.5, color='red', linestyle='--')
    ax[0].set_title(f'{title} Metrics')
    
    disp = ConfusionMatrixDisplay(metrics['confusion_matrix'],
                                display_labels=['Even', 'Odd'])
    disp.plot(ax=ax[1])
    ax[1].set_title(f'{title} Confusion Matrix')
    
    plt.tight_layout()
    plt.show()