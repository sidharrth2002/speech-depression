'''
This script assembles and runs all the models in the cluster.
It is the only script that is called.
'''

# TODO: RuntimeError: stack expects each tensor to be equal size, but got [88] at entry 0 and [90] at entry 13

# set current working directory to '/home/snag0027/speech-depression/cluster'
import os

os.chdir('/home/snag0027/speech-depression/cluster')

import argparse
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer, AutoProcessor
from dataloader.dataloadermultimodal import DaicWozDatasetWithFeatures
from training.utilities import calc_classification_metrics, compute_metrics
from models.pure_ast import multistream_ast_model
from tooling.latest_model import load_latest_checkpoint
import logging
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from config import training_config

logging.basicConfig(level=logging.INFO)

# print available gpus
logging.info(f"Available GPUs: {torch.cuda.device_count()}")
# set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='ast', choices=['ast', 'handcrafted', 'ast_handcrafted', 'ast_handcrafted_attention'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_file', type=str, default='log.csv')

args = parser.parse_args()

model = multistream_ast_model(training_config=training_config)

logging.info("Printing model to check")
logging.info(model.audio_features_model)

train_dataset = DaicWozDatasetWithFeatures(split='train')
validation_dataset = DaicWozDatasetWithFeatures(split='validation')

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

dataset_config = {
    "LOADING_SCRIPT_FILES": "/home/snag0027/speech-depression/cluster/dataloader/dataloadermultimodal.py",
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


# data collator
def data_collator(features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    batch = {}
    # TODO: x["audio_features"] is already a torch.Tensor, fix this
    logging.debug("Features: {}".format(features[0].keys()))
    for x in features:
        if len(x["audio_features"]) != 88:
            logging.info("Something's wrong")
            logging.info("Length of audio features: {}".format(len(x["audio_features"])))
            logging.info(x)
    batch["audio_features"] = torch.stack([torch.tensor(x["audio_features"]) for x in features])
    # batch["file"] = [x["file"] for x in features]
    batch["labels"] = torch.stack([torch.tensor(x["label"]) for x in features])
    # batch["audio"] = torch.stack([torch.tensor(x["audio"]) for x in features]).to(device)
    batch["input_values"] = torch.stack([torch.tensor(x["input_values"]) for x in features])
    return batch

model_path = "./trained_models/ast_multistream_5class_2"

# load model from "./trained_models/ast"
if args.model_type == 'ast':    
    if os.path.exists(model_path):
        model = load_latest_checkpoint(model_path, training_config, multistream_ast_model)
        logging.info("Loaded model from " + model_path)

    else:
        logging.info("Starting training of multistream AST model...")

        # first train the entire model for 5 epochs
        training_args = TrainingArguments(output_dir=model_path, evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=5, per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=4, eval_accumulation_steps=4, logging_steps=100, save_total_limit=3, load_best_model_at_end=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=calc_classification_metrics,
            tokenizer=feature_extractor,
            data_collator=data_collator,
        )
    
        # TODO: remove this, for debugging only
        # checks if all the layer sizes are correct
        trainer.evaluate()
        
        trainer.train()

    logging.info("Training of multistream AST model complete! Proceeding to only train the tabular classifier...")
    logging.info("Freezing all layers except audio features model... Number of trainable parameters before freezing: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # freeze only the AST
    for param in model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False

    logging.info("Freezing Complete! Number of trainable parameters after freezing: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    training_args = TrainingArguments(output_dir=model_path, evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=20, per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=4, eval_accumulation_steps=4, logging_steps=100, save_total_limit=3, load_best_model_at_end=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=calc_classification_metrics,
        tokenizer=feature_extractor,
        data_collator=data_collator,
    )

    trainer.train()

    logging.info("Finished training of multistream AST model.")

    logging.info("Evaluating model on test set...")

    results = trainer.evaluate(encoded_dataset["test"])

    logging.info("Test Set Results")
    logging.info(results)