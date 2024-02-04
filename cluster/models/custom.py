import math
from typing import Optional
from transformers import ASTFeatureExtractor, ASTModel, ASTForAudioClassification, AutoConfig
from torch import nn
import torch
import torch.nn.functional as F
from feature_processing.process_features import get_num_features
# from tabular_config import ModelArguments
# from tabular_config import TabularConfig
import logging
from config import training_config

logging.basicConfig(level=logging.INFO)

from models.layer_utils import MLP, calc_mlp_dims, glorot, hf_loss_func, zeros

class HandcraftedModelWithAudioFeatures(nn.Module):
    '''
    2D convolutions, input is MFCC features
    '''
    def __init__(self, num_classes, feature_set='mfcc', output_dim_num=100, direct_classification=True):
        super(HandcraftedModelWithAudioFeatures, self).__init__()

        logging.info("Using HandcraftedModelWithAudioFeatures")

        self.direct_classification = direct_classification

        # 4 2D convolutional layers
        # mfcc is 2d of shape (time x features)
        # egemaps is 1d of shape (1 x features)

        if feature_set == 'mfcc':
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1))
        elif feature_set == 'egemaps':        
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 1))

        # add batchnorm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # add maxpool layers
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # add dense layers

        if feature_set == 'mfcc':
            self.fc1 = nn.Linear(5888, 512)
        else:
            self.fc1 = nn.Linear(256 * 2 * 9, 512)
    
        if self.direct_classification:
            self.fc2 = nn.Linear(512, num_classes)
        else:
            self.fc2 = nn.Linear(512, output_dim_num)

    def forward(self, x):
        # torch convolution
        # swap first and second axes
        x = x.transpose(1, 2)
        # add channel dimension
        x = x.unsqueeze(1)
        logging.info("x shape: " + str(x.shape))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # maxpool
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # maxpool
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # maxpool
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.direct_classification:
            x = self.fc2(x)
            return x
        else:
            return x    

class HandcraftedModel(nn.Module):
    '''
    Classification using only handcrafted features
    '''

    def __init__(self, num_classes, num_features=450, output_dim_num=100, direct_classification=False, feature_set=training_config['feature_family']):
        super(HandcraftedModel, self).__init__()

        logging.info("Using HandcraftedModel")

        self.direct_classification = direct_classification

        # 3 convolutional layers
        # number of features is 450
        self.input_fc = nn.Linear(num_features, 450)
        
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.conv4 = nn.Conv1d(128, 256, 3)

        # add batchnorm layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # compute fc1 input size - what is the output size of the last conv layer?
        # if is09, input is 96256 
        # if egemaps, input is 20480

        if feature_set == 'is09':
            self.fc1 = nn.Linear(96256, 512)
        elif feature_set == 'egemaps':
            self.fc1 = nn.Linear(20480, 512)
        elif feature_set == 'mfcc':
            self.fc1 = nn.Linear(716800, 512)
            
        if self.direct_classification:
            self.fc2 = nn.Linear(512, num_classes)
        else:
            self.fc2 = nn.Linear(512, output_dim_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # torch convolution
        
        # x is tensor of (1, 16, 88)
        # make it (16, 1, 88)
        x = x.view(x.size(0), 1, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.direct_classification:
            x = self.fc2(x)
            return x
        else:
            return x

class ConvModel(nn.Module):
    # CNN to classify grayscale spectogram images
    def __init__(self, num_classes):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(476160, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TabularAST(ASTForAudioClassification):
    """
    The Tabular AST is by far the most unique model in this arsenal. It is a combination
    of an audio spectrogram transformer with tabular features that come from a standard 
    convolutional model that is designed to represent the low-level features that deep learning cannot.

    By default, the tabular model employs a 4-layer 1-dimensional Convolutional Neural Network.
    However, you can replace this with just about any neural network that outputs a logits vector.
    The tabular combiner will then take the output of the transformer and the output of the tabular model
    and run it through an attention-weighted sum before passing it onto the classifier.

    The paper proves the empirical improvement of model performance, when trained with tabular features.

    Args:
        hf_model_config (dict): The configuration for the Hugging Face model.

    Attributes:
        num_labels (int): The number of labels for classification.
        speech_out_dim (int): The dimension of the speech features.
        numerical_feat_dim (int): The dimension of the numerical features.
        audio_features_model (HandcraftedModel): The model for audio features.
        weight_num (nn.Parameter): The weight parameter for numerical features.
        bias_num (nn.Parameter): The bias parameter for numerical features.
        tabular_classifier (MLP): The classifier for tabular features.
        dropout (nn.Dropout): The dropout layer.
        weight_transformer (nn.Parameter): The weight parameter for the transformer.
        weight_a (nn.Parameter): The weight parameter for attention calculation.
        bias_transformer (nn.Parameter): The bias parameter for the transformer.
        bias (nn.Parameter): The bias parameter.
        negative_slope (float): The negative slope for leaky ReLU activation.
        final_out_dim (int): The final output dimension.

    Methods:
        __reset_parameters(): Resets the parameters of the model.
        tabular_combiner(speech_feats, numerical_feats): Combines speech and numerical features using attention weighted sum.
        forward(input_values, head_mask, labels, output_attentions, output_hidden_states, return_dict, audio_features, class_weights): Performs forward pass of the model.

    """

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)

        if training_config['method'] == 'regression':
            num_labels = 1
        else:
            num_labels = training_config["num_labels"]

        self.num_labels = num_labels

        dropout_prob = 0.5

        # TODO: make this a parameter
        self.speech_out_dim = 768

        # tabular combiner stuff
        # for attention sum, output dim is same as transformer output dim
        output_dim = self.speech_out_dim
        
        self.numerical_feat_dim = get_num_features(feature_set = 'egemaps')

        logging.info(f"Numerical feature dimension: {self.numerical_feat_dim}")
        logging.info(f"Speech feature dimension: {self.speech_out_dim}")

        if self.numerical_feat_dim > 0:
            if self.numerical_feat_dim > self.speech_out_dim:
                output_dim_num = self.speech_out_dim
                dims = calc_mlp_dims(
                    self.numerical_feat_dim,
                    division=self.mlp_division,
                    output_dim=output_dim_num,
                )
                
                self.num_mlp = MLP(
                    self.numerical_feat_dim,
                    output_dim_num,
                    num_hidden_lyr=len(dims),
                    dropout_prob=self.mlp_dropout,
                    return_layer_outs=False,
                    hidden_channels=dims,
                    bn=True,
                )
            else:
                output_dim_num = self.numerical_feat_dim

            self.audio_features_model = HandcraftedModel(num_classes=8, output_dim_num=output_dim_num, num_features=self.numerical_feat_dim)
            
            # TODO: make this a parameter
            self.weight_num = nn.Parameter(torch.rand((512, output_dim)))
            self.bias_num = nn.Parameter(torch.zeros(output_dim))

        combined_feat_dim = output_dim

        dims = calc_mlp_dims(combined_feat_dim, division=4,
                             output_dim=num_labels)
        self.tabular_classifier = MLP(
            input_dim=combined_feat_dim,
            output_dim=num_labels,
            num_hidden_lyr=len(dims),
            dropout_prob=dropout_prob,
            hidden_channels=dims,
            bn=True)

        self.dropout = nn.Dropout(0.2)

        self.weight_transformer = nn.Parameter(
            torch.rand(self.speech_out_dim, output_dim)
        )
        self.weight_a = nn.Parameter(torch.rand((1, output_dim + output_dim)))
        self.bias_transformer = nn.Parameter(torch.rand(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.negative_slope = 0.2
        self.final_out_dim = output_dim

        self.__reset_parameters()

    def __reset_parameters(self):
        glorot(self.weight_a)
        if hasattr(self, "weight_num"):
            glorot(self.weight_num)
            zeros(self.bias_num)
        glorot(self.weight_transformer)
        zeros(self.bias_transformer)

    def tabular_combiner(self, speech_feats, numerical_feats):
        '''
        Uses attention weighted sum: https://www.researchgate.net/profile/Martino-Mensio/publication/324877915/figure/fig12/AS:621610528686082@1525214907315/How-the-Attention-block-computes-a-weighted-sum-learning-the-weights-dynamically.png
        Key is speech features, query is tabular features
        '''
        # speech_features -> from transformer
        # tabular_features -> most probably egemaps / openSmile features
        
        # attention keyed by transformer text features
        w_text = torch.mm(speech_feats, self.weight_transformer)
        g_text = (
            (torch.cat([w_text, w_text], dim=-1) * self.weight_a)
            .sum(dim=1)
            .unsqueeze(0)
            .T
        )

        if numerical_feats.shape[1] != 0:

            # if self.numerical_feat_dim > self.speech_out_dim:
            #     numerical_feats = self.num_mlp(numerical_feats)

            w_num = torch.mm(numerical_feats, self.weight_num)
            g_num = (
                (torch.cat([w_text, w_num], dim=-1) * self.weight_a)
                .sum(dim=1)
                .unsqueeze(0)
                .T
            )
        else:
            w_num = None
            g_num = torch.zeros(0, device=g_text.device)

        alpha = torch.cat([g_text, g_num], dim=1)  # N by 3
        alpha = F.leaky_relu(alpha, 0.02)
        alpha = F.softmax(alpha, -1)
        stack_tensors = [
            tensor for tensor in [w_text, w_num] if tensor is not None
        ]
        combined = torch.stack(stack_tensors, dim=1)  # N by 3 by final_out_dim
        outputs_w_attention = alpha[:, :, None] * combined
        combined_feats = outputs_w_attention.sum(dim=1)  # N by final_out_dim

        # logging.info(f"Speech features shape: {speech_feats.shape}")
        # logging.info(f"Numerical features shape: {numerical_feats.shape}")
        # logging.info(f"Combined features shape: {combined_feats.shape}")

        return combined_feats

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
        """
        Performs the forward pass of the TabularAST model.

        Args:
            input_values (torch.Tensor, optional): The input values for the transformer.
            head_mask (torch.Tensor, optional): The head mask for the transformer.
            labels (torch.Tensor, optional): The labels for classification.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            audio_features (torch.Tensor, optional): The audio features.
            class_weights (torch.Tensor, optional): The class weights.

        Returns:
            loss (torch.Tensor): The loss value.
            logits (torch.Tensor): The logits.
            classifier_layer_outputs (List[torch.Tensor]): The outputs of the classifier layers.

        """
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

class MFCCLSTM(nn.Module):
    '''
    LSTM model using MFCC features
    
    Model consists of:
    - LSTM layer (60 hidden units)
    - Batch Normalisation
    - Dropout
    - LSTM layer (40 hidden units)
    - Batch Normalization
    - Dropout
    - LSTM layer (20 hidden units)
    - Batch Normalization
    - Dropout
    - Flatten
    - Dropout
    - Dense layer
    - Dense Layer
    - Dense Layer (output)

    dropout fraction value is 0.02%
    '''
    def __init__(self, num_classes):
        super(MFCCLSTM, self).__init__()
        self.num_classes = num_classes
        self.lstm1 = nn.LSTM(215, 60, batch_first=True)
        self.bn1 = nn.BatchNorm1d(13)
        self.dropout1 = nn.Dropout(0.02)
        self.lstm2 = nn.LSTM(60, 40, batch_first=True)
        self.bn2 = nn.BatchNorm1d(13)
        self.dropout2 = nn.Dropout(0.02)
        self.lstm3 = nn.LSTM(40, 20, batch_first=True)
        self.bn3 = nn.BatchNorm1d(13)
        self.dropout3 = nn.Dropout(0.02)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(260, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.num_classes) + ')'

# model_args = ModelArguments(
#     model_name_or_path='MIT/ast-finetuned-audioset-10-10-0.4593',
# )

# config = AutoConfig.from_pretrained(
#     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#     cache_dir=model_args.cache_dir,
# )
# tabular_config = TabularConfig(num_labels=8)
# config.tabular_config = tabular_config
# model = TabularAST(config)
# print(model)
# print(model(torch.rand(1, 7, 100), torch.rand(1, 20)))


# model = HandcraftedModel(8)
# # test model with random input
# inp = torch.rand(1, 7, 100)
# print(inp[:, 1, :].unsqueeze(1))

# print(model(torch.rand(1, 7, 100)))
# prediction = torch.argmax(model(torch.rand(1, 7, 100)), dim=1)
# print(prediction)

