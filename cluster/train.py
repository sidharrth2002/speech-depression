'''
This script assembles and runs all the models in the cluster.
It is the only script that is called.
'''
import argparse
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer
from cluster.dataloader.dataloader import DaicWozDataset
from cluster.training.utilities import compute_metrics
from models.pure_ast import model as PureModel
import logging

logging.basicConfig(level=logging.INFO)

# parameters are model_type, epochs and logging file
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='ast', choices=['ast', 'handcrafted', 'ast_handcrafted', 'ast_handcrafted_attention'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_file', type=str, default='log.csv')

args = parser.parse_args()

train_dataset = DaicWozDataset(split='train')
validation_dataset = DaicWozDataset(split='validation')

if args.model_type == 'ast':
    logging.info("Starting training of pure AST model...")
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=100)
    trainer = Trainer(
        model=PureModel,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
