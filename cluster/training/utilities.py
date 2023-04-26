import numpy as np
import evaluate
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)
from transformers import EvalPrediction

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    '''
    General metric computation
    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted"),
    }

def calc_classification_metrics(p: EvalPrediction):
    '''
    Used for custom model
    '''
    pred_labels = np.argmax(p.predictions, axis=1)
    pred_scores = softmax(p.predictions, axis=1)[:, 1]
    labels = p.label_ids
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels,
                                                                pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {'roc_auc': roc_auc_pred_score,
                'threshold': threshold,
                'pr_auc': pr_auc,
                'recall': recalls[ix].item(),
                'precision': precisions[ix].item(), 'f1': fscore[ix].item(),
                'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item()
                }
    else:
        acc = (pred_labels == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=pred_labels)
        result = {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
            "mcc": matthews_corrcoef(labels, pred_labels)
        }

    return result