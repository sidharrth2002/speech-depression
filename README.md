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

1. [The Dataloader](cluster/dataloader/dataloadermultimodal.py) - Set the `FEATURES_PATH` to the location of the data you downloaded. Note that the dataloader employs automatic caching, so feature extraction will not be executed every single time you instantiate the class. The dataloader does the following:
    - Loads the audio files and the LLDs.
    - Splits the data into train, validation and test sets.
    - Extracts audio-based features, mel-spectrograms and MFCCs from the clip.
    - Returns torch-readable dictionaries for batched training.
2. [The Tabular Model](https://github.com/sidharrth2002/speech-depression/blob/52f438da406e65e2b38d11e643fafae7673df7c2/cluster/models/custom.py#L222) - This is the primary model that we use. It is a combination of a HuggingFace 🤗 audio spectrogram transformer with tabular features that come from a standard convolutional model that is designed to represent the low-level features that deep learning cannot capture.
3. [Other Models You May Find Interesting](cluster/models/custom.py) - Other models such as pure LSTM/pure AST/pure CNNs are found in this file, each in their own class. Some results are used for the ablation study and they are presented in our paper.
4. [The Training Script](cluster/train.py) - This is the script that we use to train the models. It calls the dataloader, instantiates the relevant model, does additional data collation and uses a [Huggingface 🤗 Trainer](https://huggingface.co/transformers/main_classes/trainer.html) to train the model. The script is well-documented and should be easy to understand. 

If you are training on a HPC, you can send the job through something like this:

```bash
sbatch train_ast_multistream.py
```

### Contact

If you have any questions, please feel free to reach out to me at
```
sidharrth2002[at]gmail[dot]com
```

### How to Cite

If you use this work, please cite it as follows:

```bibtex
@inproceedings{10317141,
  author={Nagappan, Sidharrth and Lim, Chern Hong and Thimali Dharmaratne, Anuja},
  booktitle={2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
  title={Towards AST-LLDs for the Analysis of Depression in Speech Signals}, 
  year={2023},
  volume={},
  number={},
  pages={1323-1328},
  keywords={Deep learning;Analytical models;Information processing;Transformers;Depression;Convolutional neural networks;Mel frequency cepstral coefficient},
  doi={10.1109/APSIPAASC58517.2023.10317141}
}
```