<h1 align="center">Towards AST-LLDs for the Analysis of Depression in Speech Signals</h1>
<p align="center">Sidharrth Nagappan, Chern Hong Lim and Anuja Thimali Dharmaratne</p>

---

Recent advancements in deep learning allowed the deployment of large, multimodal models for depression analysis, but the increasing complexity of such models resulted in slow deployment times. This work proposes multi-stream audio-only models for depression analysis, that use transformer weights attended to by low-level descriptors (LLD) through an attention-weighted sum. It operates on the hypothesis that handcrafted feature sets will ameliorate extensive transformer pre-training. Extensive experimentation on the DAIC-WOZ test dataset shows that a combination of an audio spectrogram transformer (AST) and a Mel-frequency cepstral coefficient (MFCC) based convolutional neural network (AST-MFCC) produces the highest accuracy in our suite of models, but reports marginally lower macro F1 scores than both a naive AST and pure LLD-based models, suggesting that the injection of extra feature streams adds a sensibility element to models and limits false positives. However, the naive transformer-based and LLD-based models are surprisingly more effective at flagging depressed patients, although at the cost of an acceptable number of false positives. Our work suggests in totality that the addition of extra feature streams adds a distinct and controllable discriminating power to existing models and is able to assist lightweight models in low-data, audio-only settings.

<p align="center">
  <img src="https://github.com/sidharrth2002/speech-depression/assets/53941721/92e03794-9d21-4916-b034-b46c6d3bd9b1">
</p>

## Pre-requisites

The data used is the DAIC-WOZ dataset, collated by the University of Southern California. The dataset is available [here](http://dcapswoz.ict.usc.edu/). The dataset is not included in this repository, and you will need to download it separately after signing the agreement. If you are a part of the Monash research group, we have already signed this agreement and you can find the data in my folder `snag0027` in the Monarch HPC.

For downloading you can use [scraping.py](scraping/scrape_daic_woz.py).

## Recommended Reading Order

I understand that this repository may seem slightly overwhelming at the start. There is a lot of raw and miscellaneous code that is not too well-documented. This is my recommended reading order:

1. [The Dataloader](cluster/dataloader/dataloadermultimodal.py) - Set the `FEATURES_PATH` to the location of the data you downloaded. 
2. [The Tabular Model](cluster/models/custom.py) - This is the primary model that we use. It is a combination of an audio spectrogram transformer with tabular features that come from a standard convolutional model that is designed to represent the low-level features that deep learning cannot capture.
3. [Other Models You May Find Interesting](cluster/models/custom.py) - We have a lot of models in this repository. The `TabularAST` is the primary model, but we have a lot of other models that you may find interesting.

## Primary Model (Tabular AST)

```python
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
```

## How to Train

