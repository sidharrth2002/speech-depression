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
from training.utilities import compute_metrics
from cluster.models.get_models import get_conv_model, get_conv_model
import logging
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import librosa
import numpy as np
from config import training_config
from torch.autograd import Variable
from torchmetrics.classification import MulticlassAccuracy

logging.basicConfig(level=logging.INFO)

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='conv', choices=['ast', 'conv', 'handcrafted', 'ast_handcrafted', 'ast_handcrafted_attention'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_file', type=str, default='log.csv')
parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate the model on the test set', default=False)

args = parser.parse_args()

train_dataset = DaicWozDataset(split='train')
validation_dataset = DaicWozDataset(split='validation')

dataset_config = {
    "LOADING_SCRIPT_FILES": "/home/snag0027/speech-depression/cluster/dataloader/dataloader.py",
    "CONFIG_NAME": "daic_woz",
    "DATA_DIR": ".",
    "CACHE_DIR": "cache_daic_woz",
}

ds = load_dataset(
    dataset_config["LOADING_SCRIPT_FILES"],
    dataset_config["CONFIG_NAME"],
    data_dir=dataset_config["DATA_DIR"],
    cache_dir=dataset_config["CACHE_DIR"],
)

def prepare_dataset(batch):
    # read each audio file and generate the grayscale spectrogram using librosa
    spectograms = []
    for audio in batch["audio"]:
        # pad all audio files to 5 seconds
        if len(audio["array"]) < 80000:
            audio["array"] = np.pad(audio["array"], (0, 80000 - len(audio["array"])), 'constant')
        
        spectogram = librosa.feature.melspectrogram(y=audio["array"], sr=16000, n_fft=512, hop_length=160, n_mels=64, pad_mode='constant', win_length=400)
        # convert to grayscale
        spectogram = librosa.power_to_db(spectogram, ref=np.max)
        # ensure all spectograms are of the same size
        
        spectograms.append(spectogram)
    # create tensor and reshape to (batch_size, 1, 64, 64)
    spectograms = torch.tensor(spectograms).unsqueeze(1).float()
    batch["input_values"] = spectograms
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
metric = MulticlassAccuracy(num_classes=5)

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

if args.evaluate_only:
    logging.info("Only evaluating model on validation set")
    # evaluate the model on the test set
    model.eval()
    accuracy = 0
    total = 0

    with torch.no_grad():
        for batch in encoded_dataset["validation"]:
            opt.zero_grad()
            
            images = batch["input_values"]
            images = Variable(torch.tensor(images)).to(device)
            
            y_hat = model(images)
            loss = crit(y_hat, torch.tensor(batch["label"]))
            
            _, predicted = torch.max(y_hat.data, 1)
            total += torch.tensor(batch["label"]).size(0)
            accuracy += (predicted == torch.tensor(batch["label"])).sum().item()
            
        logging.info(f"Validation loss: {loss.item()}")
        # get accuracy for all batches
        accuracy = (accuracy / total) * 100
        logging.info(f"Validation accuracy: {accuracy}")
        
else:
    for epoch in range(100):
        logging.info(f"Epoch {epoch}")
        for batch in encoded_dataset["train"]:
            opt.zero_grad()
            
            images = batch["input_values"]
            images = Variable(torch.tensor(images))
            
            y_hat = model(images)
            loss = crit(y_hat, torch.tensor(batch["label"]))
            # calculate accuracy
            accuracy = metric(y_hat, torch.tensor(batch["label"]))
            
            loss.backward()
            opt.step()
        logging.info(f"Loss: {loss.item()}")
        
        if epoch % 10 == 0:
            with torch.no_grad():
                for batch in encoded_dataset["validation"]:
                    opt.zero_grad()
                    
                    images = batch["input_values"]
                    images = Variable(torch.tensor(images))
                    
                    y_hat = model(images)
                    loss = crit(y_hat, torch.tensor(batch["label"]))
                logging.info(f"Validation loss for epoch {epoch}: {loss.item()}")
                
                # y_hat = model(encoded_dataset["validation"]["input_values"])
                # loss = crit(y_hat, encoded_dataset["validation"]["label"])
                # print(epoch, loss.item())
                
        # save the model
        torch.save(model.state_dict(), f"models/{args.model_type}.pt")
        
        # early stopping
        if loss.item() < 0.1:
            print("Early stopping")
            break
                
    print("Done")