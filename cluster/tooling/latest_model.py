# in a model folder with different checkpoints, load and return the latest checkpoint
# for example, in the folder, there will be checkpoint-3400, checkpoint-3500, checkpoint-3600
# this function will return checkpoint-3600

import glob
import torch
import logging
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)

def load_latest_checkpoint(model_path, training_config, model_function):
    checkpoints = glob.glob(f"{model_path}/checkpoint-*")
    checkpoints.sort()
    logging.info(f"Using checkpoint {checkpoints[-1]}")
    return model_function(training_config, checkpoints[-1])