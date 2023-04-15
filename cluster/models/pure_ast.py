import logging
import os
from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor
from models.custom import ConvModel

logging.basicConfig(level=logging.INFO)

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def get_conv_model(training_config):
    model = ConvModel(
        num_classes=training_config['num_labels'],
    )
    return model

def get_model(training_config):
    if training_config['binary_classification']:
        logging.info("Using binary classification")
        num_labels = 2
    else:
        logging.info("Using multi-class classification")
        num_labels = training_config['num_labels']

    # check if /home/snag0027/speech-depression/cluster/trained_models/ast exists
    if 'model_path' in training_config and os.path.exists(training_config['model_path']):
        logging.info("Loading model from " + training_config['model_path'])
        model = ASTForAudioClassification.from_pretrained(training_config['model_path'], num_labels=num_labels, cache_dir="new_cache_dir/", problem_type="single_label_classification", ignore_mismatched_sizes=True)
        return model
    else:
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels, cache_dir="new_cache_dir/", problem_type="single_label_classification", ignore_mismatched_sizes=True)
        return model
