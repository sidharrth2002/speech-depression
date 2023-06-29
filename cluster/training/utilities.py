import logging

logging.basicConfig(level=logging.INFO)

import math
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
)
# from transformers import EvalPrediction
from config import training_config
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task():
    if training_config['num_labels'] == 2:
        return 'binary'
    else:
        return 'multiclass'

def compute_metrics(eval_pred):
    '''
    General metric computation
    '''
    import evaluate

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if len(np.unique(labels)) == 2:  # binary classification
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels),
            "f1": f1.compute(predictions=predictions, references=labels, average="macro"),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tp": tp.item(),
        }
    else:
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels),
            "f1": f1.compute(predictions=predictions, references=labels, average="macro"),
        }
    
def calc_regression_metrics(preds, labels):
    logging.info(f"Predictions: {preds}")
    logging.info(f"Labels: {labels}")
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }

def calc_classification_metrics(p):
    '''
    Used for custom model
    '''
    if training_config['method'] == 'classification':
        logging.debug("***** Running classification metrics *****")
        logging.debug("Predictions: {}".format(p.predictions[0]))
        pred_labels = np.argmax(p.predictions[0], axis=1)
        pred_scores = softmax(p.predictions[0], axis=1)[:, 1]
        labels = p.label_ids
        if len(np.unique(labels)) == 2:  # binary classification
            roc_auc_pred_score = roc_auc_score(labels, pred_scores)
            precisions, recalls, thresholds = precision_recall_curve(labels,
                                                                    pred_scores)

            f1 = f1_score(y_true=labels, y_pred=pred_labels, average=None)
            fscore = (2 * precisions * recalls) / (precisions + recalls)
            fscore[np.isnan(fscore)] = 0
            ix = np.argmax(fscore)

            # fscore = (2 * precisions * recalls) / (precisions + recalls)
            # fscore[np.isnan(fscore)] = 0
            # ix = np.argmax(fscore)

            threshold = thresholds[ix].item()
            pr_auc = auc(recalls, precisions)
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
            result = {
                'roc_auc': roc_auc_pred_score,
                'threshold': threshold,
                'pr_auc': pr_auc,
                'recall': recalls[ix].item(),
                'precision': precisions[ix].item(), 
                'f1_positive': f1[1].item(),
                'f1_negative': f1[0].item(),
                'fscore': fscore[ix].item(),
                'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item(),
                'acc': (tp.item() + tn.item()) / (tp.item() + tn.item() + fp.item() + fn.item()),
                'confusion_matrix': [[tn.item(), fp.item()], [fn.item(), tp.item()]],
            }
        else:
            acc = (pred_labels == labels).mean()
            f1 = f1_score(y_true=labels, y_pred=pred_labels, average="macro")
            result = {
                "acc": acc,
                "f1": f1,
                "acc_and_f1": (acc + f1) / 2,
            }

        logging.info("***** Classification metrics *****")
        logging.info(result)
        return result
    else:
        # regression
        logging.debug("***** Running regression metrics *****")
        logging.debug("Predictions: {}".format(p.predictions[0]))
        predictions = p.predictions[0]
        preds = np.squeeze(predictions)
        return calc_regression_metrics(preds, p.label_ids)
        
def compute_metrics_manual_loop(model, all_test_data):
    ''' 
    Compute metrics from manual torch training loop
    '''

    if training_config['method'] == 'classification':
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        acc = Accuracy(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
        f1 = F1Score(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
        precision = Precision(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
        recall = Recall(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
        confusion_matrix = ConfusionMatrix(num_classes=training_config['num_labels'], task = get_task()).to(device)

        acc.reset()
        f1.reset()
        precision.reset()
        recall.reset()
        confusion_matrix.reset()

        for batch in all_test_data:
            images = batch["input_values"]
            images = torch.autograd.Variable(torch.tensor(images))
            labels = torch.tensor(batch["label"])

            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            loss = crit(y_hat, labels)

            y_hat = torch.argmax(y_hat, dim=1)
            logging.debug(y_hat)
            logging.debug(labels)

            # count number of 0s and 1s in labels
            logging.debug(f"Number of 0s in labels: {torch.sum(labels == 0)}")
            logging.debug(f"Number of 1s in labels: {torch.sum(labels == 1)}")

            acc.update(y_hat, labels)
            f1.update(y_hat, labels)
            precision.update(y_hat, labels)
            recall.update(y_hat, labels)
            confusion_matrix.update(y_hat, labels)

        return {
            "loss": loss.item(),
            "accuracy": acc.compute().item(),
            "f1": f1.compute().item(),
            "precision": precision.compute().item(),
            "recall": recall.compute().item(),
            "confusion_matrix": confusion_matrix.compute().tolist(),
        }


    else:
        crit = torch.nn.MSELoss()
        # all_test_data is the validation dataset
        mae = MeanAbsoluteError().to(device)
        mse = MeanSquaredError().to(device)
        r2 = R2Score().to(device)

        mae.reset()
        mse.reset()
        r2.reset()

        for batch in all_test_data:
            images = batch["input_values"]
            images = torch.autograd.Variable(torch.tensor(images))
            labels = torch.tensor(batch["label"])

            images = images.to(device)
            labels = labels.to(device)

            y_hat = model(images)
            loss = crit(y_hat, labels)
            
            mae.update(y_hat, labels)
            mse.update(y_hat, labels)
            r2.update(y_hat, labels)
        
        return {
            "loss": loss.item(),
            "mae": mae.compute().item(),
            "mse": mse.compute().item(),
            # rmse is the square root of mse
            "rmse": torch.sqrt(mse.compute()).item(),
            "r2": r2.compute().item(),
        }
        