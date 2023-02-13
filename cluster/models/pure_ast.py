from transformers.models.audio_spectrogram_transformer import ASTForAudioClassification, ASTConfig
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer
from dataloader.dataloader import DaicWozDataset

dataset = DaicWozDataset()
config = ASTConfig()
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

model = ASTForAudioClassification()


# finetune model
