from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

config = ASTConfig(
    num_labels=24,
)
model = ASTForAudioClassification(config=config)
