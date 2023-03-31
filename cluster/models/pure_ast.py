import logging
from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor

logging.basicConfig(level=logging.INFO)

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def get_model(training_config):
    if training_config['binary_classification']:
        logging.info("Using binary classification")
        num_labels = 2
    else:
        logging.info("Using multi-class classification")
        num_labels = training_config['num_labels']

    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels, cache_dir="new_cache_dir/", problem_type="single_label_classification", ignore_mismatched_sizes=True)
    return model
