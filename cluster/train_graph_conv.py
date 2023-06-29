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
from models.pure_ast import get_graph_conv_model
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
import networkx as nx
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default=training_config['feature_family'] + '_graph_conv')
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
    "CACHE_DIR": "cache_daic_woz_conv_models_aug_noise_pitch_shift_speed",
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

    # pad second dimension to same length
    # pad the first and second dimension of each 2d array to the same length
    # find maximum length of first dimension
    # find the maximum length of the second dimension
    # make sure the maximum lengths do not cross 157 (OOM errors on SLURM)
    # feat is a list of lists (2d)
    # find maximum length of first dimension
    
    if training_config["feature_family"] == 'mfcc':
        max_dim_1 = 0
        max_dim_2 = 0
        for feat in batch[training_config["feature_family"]]:
            if len(feat) > max_dim_1:
                max_dim_1 = len(feat)
            for x in feat:
                if len(x) > max_dim_2:
                    max_dim_2 = len(x)

        if max_dim_2 != 215:
            max_dim_2 = 215

    for feat in batch[training_config["feature_family"]]:
        if training_config['feature_family'] == 'mfcc':            
            # pad the first and second dimension of each 2d array to the same length
            # pad the first dimension
            if len(feat) < max_dim_1:
                feat = np.pad(feat, ((0, max_dim_1 - len(feat)), (0, 0)), 'constant')
            # pad the second dimension
            if len(feat[0]) < max_dim_2:
                feat = np.pad(feat, ((0, 0), (0, max_dim_2 - len(feat[0]))), 'constant')

        # append to audio_features
        audio_features.append(feat)
    # create tensor and reshape to (batch_size, 1, 64, 64)
    if training_config['feature_family'] == 'mfcc':
        audio_features = torch.tensor(audio_features).float()
        logging.info(f"Shape of audio_features: {audio_features.shape}")
    elif training_config['feature_family'] == 'egemaps':
        audio_features = torch.tensor(audio_features).unsqueeze(1).float()
    elif training_config['feature_family'] == 'is09':
        audio_features = torch.tensor(audio_features).unsqueeze(1).float()
    else:
        audio_features = torch.tensor(audio_features).unsqueeze(1).float()
    batch["input_values"] = audio_features
    return batch

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def generate_adjacency_matrix(batch):
    '''
    Generate the adjacency matrix for each batch using "input_values"
    This will be used for graph convolutional neural networks
    '''
    # input_values is a torch tensor of is09 features
    # find the adjacency matrix for each element in the batch
    # then stack them together
    # then convert to a torch tensor

    adj_matrices = []

    for i in range(len(batch["input_values"])):
        # create an adjacency matrix for each element in the batch with the next element
        # currently i is (1 x 88)
        # have to make it square with the next element (88 x 88)
        # then normalize it

        if i == len(batch["input_values"]) - 1:
            break

        stacked = torch.stack([batch["input_values"][i], batch["input_values"][i+1]], dim=0)
        # stacked is (2 x 88)
        # convert to numpy array
        adjacency_matrix = nx.adjacency_matrix(nx.from_numpy_array(stacked.numpy()))
        # adjacency_matrix is a scipy sparse matrix
        # normalize it
        adjacency_matrix = normalize_adj(adjacency_matrix)
        # convert to torch tensor
        adjacency_matrix = torch.tensor(adjacency_matrix.todense())
        # adjacency_matrix is (88 x 88)
        # append to adj_matrices    
        adj_matrices.append(adjacency_matrix)

    batch["adj"] = torch.stack(adj_matrices)
    return batch

# encoded_train = ds["train"].map(prepare_dataset, batched=True, batch_size=4)
encoded_dataset = ds.map(prepare_dataset, batched=True, batch_size=8)
encoded_dataset = encoded_dataset.map(generate_adjacency_matrix, batched=True, batch_size=8)

# print length of each split
logging.info(f"Train length: {len(ds['train'])}")
logging.info(f"Validation length: {len(ds['validation'])}")
logging.info(f"Test length: {len(ds['test'])}")

model = get_graph_conv_model(training_config)
# check if "/home/snag0027/speech-depression/cluster/models/conv.pt" exists
if os.path.exists(f"models/{args.model_type}.pt"):
    # load the model
    model.load_state_dict(torch.load(f"models/{args.model_type}.pt"))
    logging.info("Loaded model")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# send model to GPU
model.to(device)

# fit the pytorch model to the training data
opt = torch.optim.Adam(model.parameters(), lr=0.001)
# use categorical cross entropy loss

crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

def get_task():
    if training_config['num_labels'] == 2:
        return 'binary'
    else:
        return 'multiclass'

# reshape the dataset into batches and shuffle
train_batches = []

train_data = encoded_dataset["train"]
train_data = train_data.shuffle()

for i in range(0, len(train_data), 32):
    d = train_data[i:i+32]
    train_batches.append(d)

validation_batches = []

validation_data = encoded_dataset["validation"]
validation_data = validation_data.shuffle()

for i in range(0, len(validation_data), 32):
    d = validation_data[i:i+32]
    validation_batches.append(d)

test_batches = []

test_data = encoded_dataset["test"]
test_data = test_data.shuffle()

for i in range(0, len(test_data), 16):
    d = test_data[i:i+16]
    test_batches.append(d)

encoded_dataset["train"] = train_batches

encoded_dataset["validation"] = validation_batches

encoded_dataset["test"] = test_batches

# send encoded_dataset to GPU
# encoded_dataset = encoded_dataset.to(device)

acc = Accuracy(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
f1 = F1Score(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
precision = Precision(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
recall = Recall(num_classes=training_config['num_labels'], average='macro', task = get_task()).to(device)
confusion_matrix = ConfusionMatrix(num_classes=training_config['num_labels'], task = get_task()).to(device)


def compute_metrics(model, all_test_data):
    # all_test_data is the validation dataset
    acc.reset()
    f1.reset()
    precision.reset()
    recall.reset()
    confusion_matrix.reset()

    for batch in all_test_data:
        images = batch["input_values"]
        images = Variable(torch.tensor(images))
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

all_f1 = []

for epoch in range(50):
    logging.info(f"Epoch {epoch}")
    model.train()

    for batch in encoded_dataset["train"]:
        opt.zero_grad()
        
        images = batch["input_values"]
        images = Variable(torch.tensor(images))
        labels = torch.tensor(batch["label"])

        images = images.to(device)
        labels = labels.to(device)

        # model takes parameters data and adj
        # it is a graph convolutional network

        y_hat = model(images, batch["adj"])
        # print shapes of y_hat and batch["label"]
        logging.debug(f"y_hat shape: {y_hat.shape}")
        logging.debug(f"batch['label'] shape: {labels}")
        loss = crit(y_hat, labels)
                
        loss.backward()
        opt.step()
    
    logging.info(f"Loss: {loss.item()}")
    # compute metrics
    
    model.eval()

    with torch.no_grad():
        # for batch in encoded_dataset["validation"]:
        #     opt.zero_grad()
            
        #     images = batch["input_values"]
        #     images = Variable(torch.tensor(images))
        #     images = images.to(device)
            
        #     y_hat = model(images)
        #     loss = crit(y_hat, torch.tensor(batch["label"]))
        # logging.info(f"Validation loss for epoch {epoch}: {loss.item()}")

        # compute metrics        
        metrics = compute_metrics(model, encoded_dataset["validation"])
        logging.info(f"Validation metrics for epoch {epoch}: {metrics}")
        all_f1.append(metrics['f1'])
    
    # save the model if the f1 score is better than the previous best or if it is the first epoch
    if epoch % 2:
        logging.info(f"Saving model to trained_models/{args.model_type}.pt")
        torch.save(model.state_dict(), f"trained_models/{args.model_type}.pt")

# finally evaluate on the test set
logging.info("Evaluating on test set")
model.eval()
logging.info("Loading best model from " + f"trained_models/{args.model_type}.pt")
model.load_state_dict(torch.load(f"trained_models/{args.model_type}.pt"))
logging.info("Loaded model")
with torch.no_grad():
    metrics = compute_metrics(model, encoded_dataset["test"])
    logging.info(f"Test metrics: {metrics}")

print("Done")