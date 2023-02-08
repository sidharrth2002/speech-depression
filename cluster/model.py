from typing import Optional
from transformers import ASTFeatureExtractor, ASTModel, ASTForAudioClassification, AutoConfig
from torch import nn
import torch
import torch.nn.functional as F
from tabular_config import ModelArguments
from tabular_config import TabularConfig

from layer_utils import MLP, calc_mlp_dims, glorot, hf_loss_func, zeros

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

    def __init__(self, num_classes, num_features=450):
        super(HandcraftedModel, self).__init__()

        # 3 convolutional layers
        # number of features is 450
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.conv3 = nn.Conv1d(64, 128, 3)

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
        # x = self.softmax(x)
        return x


class TabularAST(ASTForAudioClassification):
    '''
    Transformer speech features are keys that attend to tabular features
    '''

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

        self.audio_features_model = HandcraftedModel(num_classes=8)

        # tabular combiner stuff
        self.speech_out_dim = 100
        output_dim = self.speech_out_dim
        self.audio_feat_dim = 100

        if self.audio_feat_dim > 0:
            if self.audio_feat_dim > self.speech_out_dim:
                output_dim_audio = self.speech_out_dim
                dims = calc_mlp_dims(
                    self.audio_feat_dim,
                    division=4,
                    output_dim=output_dim_audio)
                self.audio_mlp = MLP(
                    self.audio_feat_dim,
                    output_dim_audio,
                    num_hidden_lyr=len(dims),
                    dropout_prob=dropout_prob,
                    hidden_channels=dims,
                    bn=True)
            else:
                output_dim_audio = self.audio_feat_dim

            self.weight_audio = nn.Parameter(
                torch.rand((output_dim_audio, output_dim)))

            self.bias_audio = nn.Parameter(torch.rand((output_dim)))

        self.weight_transformer = nn.Parameter(
            torch.rand((self.speech_out_dim, output_dim)))
        self.weight_a = nn.Parameter(torch.rand((1, output_dim + output_dim)))
        self.bias_transformer = nn.Parameter(torch.rand((output_dim)))
        self.bias = nn.Parameter(torch.rand((output_dim)))
        self.negative_slope = 0.2
        self.final_out_dim = output_dim
        self.__reset_parameters()

    def __reset_parameters(self):
        glorot(self.weight_a)
        if hasattr(self, 'weight_audio'):
            glorot(self.weight_audio)
            zeros(self.bias_audio)
        glorot(self.weight_transformer)
        zeros(self.bias_transformer)

    def tabular_combiner(self, speech_features, tabular_features):
        w_speech = torch.mm(speech_features, self.weight_transformer)
        g_speech = (torch.cat([w_speech, w_speech], dim=-1)
                    * self.weight_a).sum(dim=1).unsqueeze(0).T

        if tabular_features.shape[1] != 0:
            if self.audio_feat_dim > self.speech_out_dim:
                tabular_features = self.audio_mlp(tabular_features)
            w_audio = torch.mm(tabular_features, self.weight_audio)
            g_audio = (torch.cat([w_speech, w_audio], dim=-1)
                       * self.weight_a).sum(dim=1).unsqueeze(0).T
        else:
            w_audio = None
            g_audio = torch.zeros(0, device=g_audio.device)

        alpha = torch.cat([g_speech, g_audio], dim=1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, -1)
        stack_tensors = [tensor for tensor in [
            w_speech, w_audio] if tensor is not None]
        combined = torch.stack(stack_tensors, dim=1)
        outputs_with_attention = alpha[:, :, None] * combined
        combined_feats = outputs_with_attention.sum(dim=1)

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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        audio_features = self.audio_features_model(audio_features)
        combined_feats = self.tabular_combiner(
            pooled_output, audio_features)
        loss, logits, classifier_layer_outputs = hf_loss_func(
            combined_feats, self.tabular_classifier, labels, self.num_labels, class_weights)

        return loss, logits, classifier_layer_outputs


model_args = ModelArguments(
    model_name_or_path='MIT/ast-finetuned-audioset-10-10-0.4593',
)

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
tabular_config = TabularConfig(num_labels=8)
config.tabular_config = tabular_config
model = TabularAST(config)
print(model)
# print(model(torch.rand(1, 7, 100), torch.rand(1, 20)))


# model = HandcraftedModel(8)
# # test model with random input
# inp = torch.rand(1, 7, 100)
# print(inp[:, 1, :].unsqueeze(1))

# print(model(torch.rand(1, 7, 100)))
# prediction = torch.argmax(model(torch.rand(1, 7, 100)), dim=1)
# print(prediction)


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
