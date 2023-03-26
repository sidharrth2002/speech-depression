from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

model = ASTForAudioClassification(config=config)


# finetune model
