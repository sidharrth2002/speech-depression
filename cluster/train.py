'''
This script assembles and runs all the models in the cluster.
It is the only script that is called.
'''

# set current working directory to '/home/snag0027/speech-depression/cluster'
import os
os.chdir('/home/snag0027/speech-depression/cluster')

import argparse
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer, AutoProcessor
from dataloader.dataloader import DaicWozDataset
from training.utilities import compute_metrics
from models.pure_ast import get_model
import logging
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from config import training_config

logging.basicConfig(level=logging.INFO)

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='ast', choices=['ast', 'handcrafted', 'ast_handcrafted', 'ast_handcrafted_attention'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_file', type=str, default='log.csv')

args = parser.parse_args()

train_dataset = DaicWozDataset(split='train')
validation_dataset = DaicWozDataset(split='validation')

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

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
    audio_arrays = [x["array"] for x in batch["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
    return inputs

# encoded_train = ds["train"].map(prepare_dataset, batched=True, batch_size=4)
encoded_dataset = ds.map(prepare_dataset, batched=True, batch_size=8)

# print length of each split
logging.debug("Train length: ", len(ds["train"]))
logging.debug("Validation length: ", len(ds["validation"]))
logging.debug("Test length: ", len(ds["test"]))

model = get_model(training_config=training_config)

# load model from "./trained_models/ast"
if args.model_type == 'ast':
    logging.info("Starting training of pure AST model...")
    training_args = TrainingArguments(output_dir="./trained_models/ast_5class", evaluation_strategy="epoch", num_train_epochs=4, per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=16, eval_accumulation_steps=16, logging_steps=6, save_steps=6)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    trainer.train()
    logging.info("Finished training of pure AST model.")
    logging.info("Saved in ./trained_models/ast")