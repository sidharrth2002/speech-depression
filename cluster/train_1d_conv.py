'''
This script assembles and runs all the models in the cluster.
It is the only script that is called.
'''

# set current working directory to '/home/snag0027/speech-depression/cluster'
import os
os.chdir('/home/snag0027/speech-depression/cluster')

import argparse
from transformers import AutoFeatureExtractor, AutoProcessor
from dataloader.dataloader import DaicWozDataset
from models.pure_ast import get_conv_model, get_conv_model
import logging
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import librosa
import numpy as np
from config import training_config
from torch.autograd import Variable
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

logging.basicConfig(level=logging.INFO)

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default=training_config['feature_family'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_file', type=str, default='log.csv')
parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate the model on the test set', default=False)

args = parser.parse_args()

logging.info(f"Using model type {args.model_type}")

train_dataset = DaicWozDataset(split='train')
validation_dataset = DaicWozDataset(split='validation')

dataset_config = {
    "LOADING_SCRIPT_FILES": "/home/snag0027/speech-depression/cluster/dataloader/dataloadermultimodal.py",
    "CONFIG_NAME": "daic_woz",
    "DATA_DIR": ".",
    "CACHE_DIR": "cache_daic_woz_conv_models",
}

ds = load_dataset(
    dataset_config["LOADING_SCRIPT_FILES"],
    dataset_config["CONFIG_NAME"],
    data_dir=dataset_config["DATA_DIR"],
    cache_dir=dataset_config["CACHE_DIR"],
)

def prepare_dataset(batch):
    # read each audio file and generate the grayscale spectrogram using librosa
    audio_features = []
    # feature_family specifies the feature set to use
    for feat in batch[training_config["feature_family"]]:
        # pad all audio files to 5 seconds
        audio_features.append(feat)
    # create tensor and reshape to (batch_size, 1, 64, 64)
    audio_features = torch.tensor(audio_features).unsqueeze(1).float()
    batch["input_values"] = audio_features
    return batch
    
# encoded_train = ds["train"].map(prepare_dataset, batched=True, batch_size=4)
encoded_dataset = ds.map(prepare_dataset, batched=True, batch_size=8)

# print length of each split
logging.info(f"Train length: {len(ds['train'])}")
logging.info(f"Validation length: {len(ds['validation'])}")
logging.info(f"Test length: {len(ds['test'])}")

model = get_conv_model(training_config=training_config)
# check if "/home/snag0027/speech-depression/cluster/models/conv.pt" exists
if os.path.exists(f"models/{args.model_type}.pt"):
    # load the model
    model.load_state_dict(torch.load(f"models/{args.model_type}.pt"))
    logging.info("Loaded model")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# fit the pytorch model to the training data
opt = torch.optim.Adam(model.parameters(), lr=0.001)
# use categorical cross entropy loss
crit = torch.nn.CrossEntropyLoss()

def get_task():
    if training_config['num_labels'] == 2:
        return 'binary'
    else:
        return 'multiclass'

# reshape the data to batches of 8
train_batches = []
for i in range(0, len(encoded_dataset["train"]), 32):
    train_batches.append(encoded_dataset["train"][i:i+32])

validation_batches = []
for i in range(0, len(encoded_dataset["validation"]), 32):
    validation_batches.append(encoded_dataset["validation"][i:i+32])

encoded_dataset["train"] = train_batches
encoded_dataset["validation"] = validation_batches

# send encoded_dataset to GPU
encoded_dataset = encoded_dataset.to(device)

acc = Accuracy(num_classes=training_config['num_labels'], average='macro', task = get_task())
f1 = F1Score(num_classes=training_config['num_labels'], average='macro', task = get_task())
precision = Precision(num_classes=training_config['num_labels'], average='macro', task = get_task())
recall = Recall(num_classes=training_config['num_labels'], average='macro', task = get_task())
confusion_matrix = ConfusionMatrix(num_classes=training_config['num_labels'], task = get_task())


def compute_metrics(all_test_data):
    # all_test_data is the validation dataset
    acc.reset()
    f1.reset()
    precision.reset()
    recall.reset()
    confusion_matrix.reset()

    for batch in all_test_data:
        images = batch["input_values"]
        images = Variable(torch.tensor(images))
        
        y_hat = model(images)
        loss = crit(y_hat, torch.tensor(batch["label"]))
        
        acc.update(y_hat, torch.tensor(batch["label"]))
        f1.update(y_hat, torch.tensor(batch["label"]))
        precision.update(y_hat, torch.tensor(batch["label"]))
        recall.update(y_hat, torch.tensor(batch["label"]))
        confusion_matrix.update(y_hat, torch.tensor(batch["label"]))

    return {
        "accuracy": acc.compute(),
        "f1": f1.compute(),
        "precision": precision.compute(),
        "recall": recall.compute(),
        "confusion_matrix": confusion_matrix.compute().tolist()
    }

all_f1 = []

for epoch in range(50):
    logging.info(f"Epoch {epoch}")
    for batch in encoded_dataset["train"]:
        opt.zero_grad()
        
        images = batch["input_values"]
        images = Variable(torch.tensor(images))
        
        y_hat = model(images)
        loss = crit(y_hat, torch.tensor(batch["label"]))
                
        loss.backward()
        opt.step()
    
    logging.info(f"Loss: {loss.item()}")
    # compute metrics
    
    model.eval()

    with torch.no_grad():
        for batch in encoded_dataset["validation"]:
            opt.zero_grad()
            
            images = batch["input_values"]
            images = Variable(torch.tensor(images))
            
            y_hat = model(images)
            loss = crit(y_hat, torch.tensor(batch["label"]))
        logging.info(f"Validation loss for epoch {epoch}: {loss.item()}")

        # compute metrics        
        metrics = compute_metrics(encoded_dataset["validation"])
        logging.info(f"Validation metrics for epoch {epoch}: {metrics}")
        all_f1.append(metrics['f1'])
    
    # save the model if the f1 score is better than the previous best or if it is the first epoch
    if epoch == 0 or metrics['f1'] > max(all_f1):
        logging.info(f"Saving model to trained_models/{args.model_type}.pt because f1 score of {metrics['f1']} is better than previous best of {max(all_f1)}")
        torch.save(model.state_dict(), f"trained_models/{args.model_type}.pt")

    # # early stopping
    # if loss.item() < 0.1:
    #     print("Early stopping")
    #     break
            
print("Done")