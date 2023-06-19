import logging
import os

from feature_processing.process_features import get_num_features
from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor
from models.custom import HandcraftedModelWithAudioFeatures, HandcraftedModel, TabularAST, GraphCNN

logging.basicConfig(level=logging.INFO)

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def get_1d_conv_model(training_config):
    model = HandcraftedModel(num_classes=training_config['num_labels'], output_dim_num=training_config['num_labels'], direct_classification=True)
    return model

def get_conv_model(training_config):
    if training_config['feature_family'] in ['egemaps', 'is09']:
        model = HandcraftedModel(num_classes=training_config['num_labels'], output_dim_num=training_config['num_labels'], direct_classification=True, feature_set=training_config['feature_family'])
        return model
    else:
        model = HandcraftedModelWithAudioFeatures(
            num_classes=training_config['num_labels'],
            feature_set=training_config['feature_family'],
        )
        return model
    
def get_graph_conv_model(training_config):
    return GraphCNN(num_features=get_num_features(training_config['feature_family']))

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

def multistream_ast_model(training_config, model_path=None):
    # model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
    if training_config['binary_classification']:
        logging.info("Using binary classification")
        num_labels = 2
    else:
        logging.info("Using multi-class classification")
        num_labels = training_config['num_labels']
    
    if model_path is not None and os.path.exists(model_path):
        logging.info("Loading model from " + model_path)
        model = TabularAST.from_pretrained(model_path, num_labels=num_labels, cache_dir="new_cache_dir/", problem_type="single_label_classification", ignore_mismatched_sizes=True)
        return model

    else:
        model = TabularAST.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels, cache_dir="new_cache_dir/", problem_type="single_label_classification", ignore_mismatched_sizes=True)
        return model
