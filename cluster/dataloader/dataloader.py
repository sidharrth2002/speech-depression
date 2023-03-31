# Lint as: python3
"""DAIC-WOZ Dataset."""
"""Main file for DAIC-WOZ dataset"""

'''
Custom huggingface dataloader to return audio and labels
'''
import json
import logging
import os
import datasets
import pandas as pd
import librosa
import soundfile
from config import training_config

logging.basicConfig(level=logging.INFO)

class DaicWozConfig(datasets.BuilderConfig):
    """
    Builder config for DAIC-WOZ dataset
    """

    def __init__(self, **kwargs):
        super(DaicWozConfig, self).__init__(
            version=datasets.Version("0.0.1", ""), **kwargs)

'''
0-4 -> Minimal depression
5-9 -> Mild depression
10-14 -> Moderate depression
15-19 -> Moderately severe depression
20-24 -> Severe depression
'''
def place_value_in_bin(value, put_in_bin=False):
    if put_in_bin:
        if value <= 4:
            return 0
        elif value <= 9:
            return 1
        elif value <= 14:
            return 2
        elif value <= 19:
            return 3
        else:
            return 4
    else:
        return value

class DaicWozDataset(datasets.GeneratorBasedBuilder):
    '''
    Return audio, features, and labels for DAIC-WOZ dataset
    '''
    BUILDER_CONFIGS = [
        DaicWozConfig(
            name="daic_woz",
            description="DAIC-WOZ dataset",
        ),
    ]
    
    BINARY_CLASSIFICATION = training_config['binary_classification']


    def __init__(self, *args, writer_batch_size=None, split='train', **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.split = split
        self.dataset_path = '/home/snag0027/daic_woz'


    def _info(self):
        return datasets.DatasetInfo(
            description="DAIC-WOZ dataset",
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16_000),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=("file", "label"),
        )

    def _split_generators(self, dl_manager):
        # already downloaded to a specified directory
        data_dir_train = f'{self.dataset_path}/splits/train'
        data_dir_val = f'{self.dataset_path}/splits/val'
        data_dir_test = f'{self.dataset_path}/splits/test'
        if self.split == 'train':
            self.data_dir = data_dir_train
        elif self.split == 'val':
            self.data_dir = data_dir_val
        else:
            self.data_dir = data_dir_test
        if self.config.name == 'daic_woz':
            train_splits = [
                datasets.SplitGenerator(
                    name="train", gen_kwargs={"name": "train"}
                ),
                datasets.SplitGenerator(
                    name="validation", gen_kwargs={"name": "validation"}
                ),
                datasets.SplitGenerator(
                    name="test", gen_kwargs={"name": "test"}
                ),
            ]
        return train_splits

    def _generate_examples(self, name):
        key = 0
        examples = []

        if name == "train":
            audio_dir = f'{self.dataset_path}/splits/train'
        elif name == "validation":
            audio_dir = f'{self.dataset_path}/splits/val'
        else:
            audio_dir = f'{self.dataset_path}/splits/test'

        if not os.path.exists(audio_dir):
            raise FileNotFoundError(
                f"{audio_dir} does not exist. Make sure you insert the correct path to the audio directory."
            )

        label_file = os.path.join(self.dataset_path, "labels.csv")

        label_file = pd.read_csv(label_file)
        # set Participant_ID as index
        label_file = label_file.set_index("Participant_ID")
        label_file['PHQ8_Score'] = label_file['PHQ8_Score'].astype('int')
        label_file['PHQ8_Binary'] = label_file['PHQ8_Binary'].astype('int')

        for folder in os.listdir(audio_dir):
            for file in os.listdir(os.path.join(audio_dir, folder)):
                if 'chunk' in file:
                    # delete the file
                    os.remove(os.path.join(audio_dir, folder, file))

        for folder in os.listdir(audio_dir):
            print(folder)
            # check if folder is a directory and that the first 3 characters are numeric
            if os.path.isdir(os.path.join(audio_dir, folder)) and folder[:3].isnumeric():
                if training_config['break_audio_into_chunks']:
                    for file in os.listdir(os.path.join(audio_dir, folder)):
                        if file.endswith("PARTICIPANT_merged.wav") and not file.startswith('._'):
                            # break audio into chunks of 8 seconds each
                            # then save each chunk as a separate file
                            audio, sr = librosa.load(os.path.join(audio_dir, folder, file), sr=16000)
                            
                            # remove pauses in audio
                            audio, _ = librosa.effects.trim(audio, top_db=20, frame_length=2, hop_length=500)
                            
                            # break audio into chunks of 8 seconds each, with no silence
                            # then save each chunk as a separate file
                            for i in range(0, len(audio), 8 * sr):
                                chunk = audio[i:i + 8 * sr]
                                # save chunk
                                # check if chunk is not empty and file does not already exist
                                if chunk.size > 0 and not os.path.exists(os.path.join(audio_dir, folder, f"{file[:-4]}_chunk_{i}.wav")):
                                    soundfile.write(os.path.join(audio_dir, folder, f"{file[:-4]}_chunk_{i}.wav"), chunk, sr)
                                                        
                for file in os.listdir(os.path.join(audio_dir, folder)):
                    if training_config['break_audio_into_chunks']:
                        # only use chunk files
                        if not file.startswith('._') and 'chunk' in file:
                            if self.BINARY_CLASSIFICATION:
                                logging.debug("Using binary classification")
                                examples.append(
                                    {
                                        "file": os.path.join(audio_dir, folder, file),
                                        "audio": os.path.join(audio_dir, folder, file),
                                        # from 300P to 300
                                        "label": int(label_file.loc[int(folder[:3])]["PHQ8_Binary"])
                                    }
                                )
                            else:
                                examples.append(
                                    {
                                        "file": os.path.join(audio_dir, folder, file),
                                        "audio": os.path.join(audio_dir, folder, file),
                                        # from 300P to 300
                                        "label": place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True),
                                    }
                                )
                    else:
                        if file.endswith("PARTICIPANT_merged.wav") and not file.startswith('._'):                        
                            if self.BINARY_CLASSIFICATION:
                                logging.debug("Using binary classification")
                                examples.append(
                                    {
                                        "file": os.path.join(audio_dir, folder, file),
                                        "audio": os.path.join(audio_dir, folder, file),
                                        # from 300P to 300
                                        "label": int(label_file.loc[int(folder[:3])]["PHQ8_Binary"])
                                    }
                                )
                            else:
                                examples.append(
                                    {
                                        "file": os.path.join(audio_dir, folder, file),
                                        "audio": os.path.join(audio_dir, folder, file),
                                        # from 300P to 300
                                        "label": place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True),
                                    }
                                )

        for example in examples:
            yield key, {**example}
            key += 1
        examples = []

    def get_data(self, name):
        key = 0
        examples = []

        audio_dir = self.dataset_path

        if not os.path.exists(audio_dir):
            raise FileNotFoundError(
                f"{audio_dir} does not exist. Make sure you insert the correct path to the audio directory."
            )

        label_file = os.path.join(audio_dir, "labels.csv")

        label_file = pd.read_csv(label_file)
        # set Participant_ID as index
        label_file = label_file.set_index("Participant_ID")

        for folder in os.listdir(audio_dir):
            # check if folder is a directory and that the first 3 characters are numeric
            if os.path.isdir(os.path.join(audio_dir, folder)) and folder[:3].isnumeric():
                for file in os.listdir(os.path.join(audio_dir, folder)):
                    if file.endswith("PARTICIPANT_merged.wav") and not file.startswith('._'):
                        if self.BINARY_CLASSIFICATION:
                            examples.append(
                                {
                                    "file": os.path.join(audio_dir, folder, file),
                                    "audio": os.path.join(audio_dir, folder, file),
                                    # from 300P to 300
                                    "label": int(label_file.loc[int(folder[:3])]["PHQ8_Binary"])
                                }
                            )
                        else:
                            examples.append(
                                {
                                    "file": os.path.join(audio_dir, folder, file),
                                    "audio": os.path.join(audio_dir, folder, file),
                                    # from 300P to 300
                                    "label": place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True),
                                }
                            )

        return examples


    def __len__(self):
        return len(self.get_data(self.split))
    