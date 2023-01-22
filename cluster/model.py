from typing import Optional
from transformers import ASTFeatureExtractor, ASTModel, ASTForAudioClassification
from torch import nn
import torch

from cluster.layer_utils import MLP, calc_mlp_dims, hf_loss_func

processor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")

'''
1. Prosodic features
2. Spectral features
3. Mel frequency cepstral coefficients (MFCC)
4. Linear prediction cepstral coefficients (LPCC)
5. Gammatone frequency cepstral coefficients (GFCC)
6. Voice quality features
7. Teager energy operator (TEO)
Build a convolutional model that takes the above features as input and outputs the PHQ-8 score
'''


class HandcraftedModel(nn.Module):
    '''
    Classification using only handcrafted features
    '''

    def __init__(self, num_classes):
        super(HandcraftedModel, self).__init__()
        self.prosodic_conv = nn.Conv1d(1, 2, 3)
        self.spectral_conv = nn.Conv1d(1, 2, 3)
        self.mfcc_conv = nn.Conv1d(1, 2, 3)
        self.lpcc_conv = nn.Conv1d(1, 2, 3)
        self.gfcc_conv = nn.Conv1d(1, 2, 3)
        self.voice_conv = nn.Conv1d(1, 2, 3)
        self.teo_conv = nn.Conv1d(1, 2, 3)
        # compute fc1 input size
        self.fc1 = nn.Linear(1372, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # torch multi-channel convolution
        prosodic = self.prosodic_conv(x[:, 0, :].unsqueeze(1))
        spectral = self.spectral_conv(x[:, 1, :].unsqueeze(1))
        mfcc = self.mfcc_conv(x[:, 2, :].unsqueeze(1))
        lpcc = self.lpcc_conv(x[:, 3, :].unsqueeze(1))
        gfcc = self.gfcc_conv(x[:, 4, :].unsqueeze(1))
        voice = self.voice_conv(x[:, 5, :].unsqueeze(1))
        teo = self.teo_conv(x[:, 6, :].unsqueeze(1))
        x = torch.cat((prosodic, spectral, mfcc, lpcc, gfcc, voice, teo), 1)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class TabularAST(ASTForAudioClassification):
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        combined_feat_dim = 100
        num_labels = 8
        dropout_prob = 0.5
        dims = calc_mlp_dims(combined_feat_dim, division=4,
                             output_dim=num_labels)
        self.tabular_classifier = MLP(
            combined_feat_dim,
            num_labels,
            num_hidden_lyr=len(dims),
            dropout_prob=dropout_prob,
            hidden_channels=dims,
            bn=True)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_features: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        outputs = self.audio_spectrogram_transformer(
            input_values,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(
            pooled_output, audio_features)
        loss, logits, classifier_layer_outputs = hf_loss_func(
            combined_feats, self.tabular_classifier, labels, self.num_labels, class_weights)

        return loss, logits, classifier_layer_outputs


model = HandcraftedModel(8)
# test model with random input
inp = torch.rand(1, 7, 100)
print(inp[:, 1, :].unsqueeze(1))

print(model(torch.rand(1, 7, 100)))
prediction = torch.argmax(model(torch.rand(1, 7, 100)), dim=1)
print(prediction)


class ASTConcat(nn.Module):
    def __init__(self, num_classes):
        super(ASTConcat, self).__init__()
        self.ast = model
        self.fc1 = nn.Linear(1280 + 20, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, mfcc):
        x = self.ast(x)
        x = torch.cat((x, mfcc), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.ast.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

# AST + MFCC features (combine with attention)


class ASTHandcraftedAttention(nn.Module):
    def __init__(self, num_classes):
        super(ASTHandcraftedAttention, self).__init__()
        self.ast = model
        self.fc1 = nn.Linear(1280 + 20, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def attention_weighted_sum(self, f1, f2):
        f1 = torch.softmax(f1, 1)
        f2 = torch.softmax(f2, 1)
        f1 = torch.mul(f1, f2)
        return f1

    def forward(self, x, mfcc):
        x = self.ast(x)
        x = self.attention_weighted_sum(x, mfcc)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
