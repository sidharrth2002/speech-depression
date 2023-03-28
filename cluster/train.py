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
from models.pure_ast import model as PureModel
import logging
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

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

print("Printing type")
print(ds["validation"])
print(ds["test"])

def prepare_dataset(batch):
    audio_arrays = [x["array"] for x in batch["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
    # batch["input_values"] = [feature_extractor(x["array"], sampling_rate=16000, return_tensors="pt", padding=True) for x in audio]
    # batch["labels"] = [int(x) for x in batch["label"]]
    return inputs

# encoded_train = ds["train"].map(prepare_dataset, batched=True, batch_size=4)
print(ds["validation"].features)
encoded_dataset = ds.map(prepare_dataset, batched=True, batch_size=4)

# print(ds["train"]["label"])
# ds["train"]["label"] = [float(x) for x in ds["train"]["label"]]
# ds["validation"]["label"] = [float(x) for x in ds["validation"]["label"]]
# ds["test"]["label"] = [float(x) for x in ds["test"]["label"]]

# # 80-20-20 split
# train_testval = ds["train"].train_test_split(test_size=0.2, seed=42)
# test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)
# ds = DatasetDict({"train": train_testval["train"], "validation": test_val["test"], "test": test_val["train"]})

# print length of each split
print("Train length: ", len(ds["train"]))
print("Validation length: ", len(ds["validation"]))
print("Test length: ", len(ds["test"]))

print("Printing sample")
print(type(ds["train"][0]))
print(type(ds["train"][0]["audio"]["array"]))

# encoded_train = ds["train"].map(lambda x: print(x), batched=True)

# encoded_train = ds["train"].map(lambda x: processor(x["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True), batched=True)
# encoded_validation = ds["validation"].map(lambda x: processor(x["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True), batched=True)

@dataclass
class DataCollator:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        input_features = [{
            "input_features": feature["input_values"],
        } for feature in features]

        batch = self.processor.pad(input_features, return_tensors="pt")

        labels = [feature["labels"] for feature in features]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

data_collator = DataCollator(processor=processor)

if args.model_type == 'ast':
    logging.info("Starting training of pure AST model...")
    training_args = TrainingArguments(output_dir="./trained_models/ast", evaluation_strategy="epoch", num_train_epochs=4, per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4, eval_accumulation_steps=4, logging_steps=6)
    trainer = Trainer(
        model=PureModel,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    trainer.train()
    logging.info("Finished training of pure AST model.")
    logging.info("Saved in ./trained_models/ast")