import json
import logging
import os
import random
import subprocess

from dataloader.utils import place_value_in_bin
import datasets
import pandas as pd
import opensmile
from config import training_config
import librosa
from feature_processing.process_features import process_features, get_num_features
from tooling.augment import augment_audio
import soundfile
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)

class DaicWozConfig(datasets.BuilderConfig):
    """
    Builder config for DAIC-WOZ dataset
    """

    def __init__(self, **kwargs):
        super(DaicWozConfig, self).__init__(
            version=datasets.Version("0.0.1", ""), **kwargs)

class DaicWozDatasetWithFeatures(datasets.GeneratorBasedBuilder):
    '''
    Return only the audio and labels
    '''
    BUILDER_CONFIGS = [
        DaicWozConfig(
            name="daic_woz",
            description="DAIC-WOZ dataset with features",
        ),
    ]
    FEATURES_PATH = '/home/snag0027/features.csv'
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
                    "audio_features": datasets.features.Sequence(datasets.Value("float32")),
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


    def _balance_data(self, data):
        logging.info("Balancing data...")
        # find how many unique labels there are
        unique_labels = set([x['label'] for x in data])
        # find the number of examples for each label
        num_examples = {}
        for label in unique_labels:
            num_examples[label] = len([x for x in data if x['label'] == label])
        

        logging.info(f"Number of examples for each label: {num_examples}")
        logging.info(f"Total number of examples: {len(data)}")
        logging.info(f"Number of unique labels: {len(unique_labels)}")
        
        # augment all the data until all labels have the same number of examples
        # augment the audio using the augment_audio function

        # how many extra examples in each minority class to balance the data
        needed_per_class = {}
        for label in unique_labels:
            needed_per_class[label] = max(num_examples.values()) - num_examples[label]

        # augment the data
        for label in needed_per_class:
            # find all examples with this label
            examples = [x for x in data if x['label'] == label]
            # randomly select needed_per_class[label] examples only if needed_per_class[label] is less than the number of examples
            # otherwise, you can select an example multiple times
            if needed_per_class[label] < len(examples):
                examples = random.sample(examples, needed_per_class[label])

                # augment each example
                for example in examples:
                    # check if the audio is not empty
                    if len(example['audio']) == 0:
                        logging.info(f"Audio for {example['file']} is empty, so skipping")
                        continue
                    # augment the audio
                    augmented_audio = augment_audio(example['file'])
                    # save the augmented audio

                    file_path = example["file"]
                    folder = file_path.split("/")[-2]
                    file_name = file_path.split("/")[-1]

                    # number of times this file has already been augmented
                    num_augmented = len([x for x in os.listdir(os.path.join(self.data_dir, folder)) if x.startswith(file_name[:-4]) and x.endswith(".wav")])

                    soundfile.write(os.path.join(self.data_dir, folder, f"{file_name[:-4]}_augmented_{num_augmented + 1}.wav"), augmented_audio[0], 16000)
                    # add the augmented audio to the data
                    data.append(
                        {
                            "file": example['file'],
                            "audio": augmented_audio,
                            "audio_features": self._extract_audio_features(self.data_dir, folder, f"{file_name[:-4]}_augmented_{num_augmented + 1}.wav"),
                            "label": example['label']
                        }
                    )
            else:
                # if there are not enough examples, then augment the examples until there are enough
                while len(examples) < needed_per_class[label]:
                    # augment each example
                    for example in examples:
                        if len(example['audio']) == 0:
                            logging.info(f"Audio for {example['file']} is empty, so skipping")
                            continue

                        # augment the audio
                        augmented_audio = augment_audio(example['file'])
                        # save the augmented audio

                        file_path = example["file"]
                        folder = file_path.split("/")[-2]
                        file_name = file_path.split("/")[-1]

                        # number of times this file has already been augmented
                        num_augmented = len([x for x in os.listdir(os.path.join(self.data_dir, folder)) if x.startswith(file_name[:-4]) and x.endswith(".wav")])

                        a = example["audio"]
                        logging.debug(f"Before augmentation: {a}")
                        logging.debug(f"After augmentation: {augmented_audio}")

                        soundfile.write(os.path.join(self.data_dir, folder, f"{file_name[:-4]}_augmented_{num_augmented + 1}.wav"), augmented_audio[0], 16000)
                        # add the augmented audio to the data
                        data.append(
                            {
                                "file": example['file'],
                                "audio": augmented_audio,
                                "audio_features": self._extract_audio_features(self.data_dir, folder, f"{file_name[:-4]}_augmented_{num_augmented + 1}.wav"),
                                "label": example['label']
                            }
                        )
        # shuffle the data
        random.shuffle(data)
        
        # find the number of examples for each label
        num_examples = {}
        for label in unique_labels:
            num_examples[label] = len([x for x in data if x['label'] == label])
        
        logging.info(f"Number of examples for each label after balancing: {num_examples}")
        logging.info(f"Total number of examples after balancing: {len(data)}")
        logging.info(f"Number of unique labels after balancing: {len(unique_labels)}")

        return data

    def _extract_audio_features(self, audio_dir, folder, file):
        if (file.endswith(".wav")) and (not file.startswith('._')):
            if os.path.exists(os.path.join(audio_dir, folder, file[:-4]) + ".csv"):
                logging.info(f"eGeMAPSv02 features for {file} already extracted, so skipping")
            else:
                # if features have not already been extracted, extract them
                logging.info(f"Extracting eGeMAPSv02 features for {file}...")
                extraction_command = f"SMILExtract -C /home/snag0027/speech-depression/cluster/config/egemaps/v02/eGeMAPSv02.conf -I {os.path.join(audio_dir, folder, file)} -O {os.path.join(audio_dir, folder, file[:-4])}.csv"
                subprocess.call(extraction_command, shell=True, cwd='/home/snag0027/speech-depression/cluster/opensmile/bin')
            
            # sometimes, smilextract fails to extract
            # in this case, we will just use zeros as the features (not a good idea)
            if os.path.exists(os.path.join(audio_dir, folder, file[:-4]) + ".csv"):
                audio_features = process_features(os.path.join(audio_dir, folder, file[:-4]) + ".csv")
            else:
                logging.info(f"Could not extract eGeMAPSv02 features for {file}")
                num_egemaps_features = get_num_features('egemaps')
                # create a torch tensor of zeros
                # audio_features = torch.tensor([0.1] * num_egemaps_features).float()
                audio_features = [0.111] * num_egemaps_features
        return audio_features

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

        features = pd.read_csv(self.FEATURES_PATH)
        features = features.set_index("Participant_ID")

        label_file = os.path.join(self.dataset_path, "labels.csv")

        label_file = pd.read_csv(label_file)
        # set Participant_ID as index
        label_file = label_file.set_index("Participant_ID")
        label_file['PHQ8_Score'] = label_file['PHQ8_Score'].astype('int')
                   
        chunk_size = training_config['chunk_size']
             
        for folder in os.listdir(audio_dir):
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
                            
                            # break audio into chunks of chunk_size seconds each, with no silence
                            # then save each chunk as a separate file
                            chunks = {}
                            for i in range(0, len(audio), chunk_size * sr):
                                chunk = audio[i:i + chunk_size * sr]
                                # save chunk
                                # check if chunk is not empty and file does not already exist
                                if chunk.size > 0 and not os.path.exists(os.path.join(audio_dir, folder, f"{file[:-4]}_chunk_{i}.wav")):
                                    chunks[f"{file[:-4]}_chunk_{i}.wav"] = chunk

                            # # randomly select num_chunks from chunks
                            # if len(chunks) > num_chunks:
                            #     chunks = random.sample(list(chunks.items()), num_chunks)
                            # else:
                            #     chunks = list(chunks.items())
                            
                            # save chunks
                            chunks = list(chunks.items())
                            
                            for chunk in chunks:
                                print(chunk[0])
                                soundfile.write(os.path.join(audio_dir, folder, chunk[0]), chunk[1], sr)
                                                            
                # For debugging in the log file
                logging.info(f"Are we breaking audio into chunks? {training_config['break_audio_into_chunks']}")
                logging.info(f"Chunk size: {chunk_size}")
                logging.info(f"Are we using binary classification? {self.BINARY_CLASSIFICATION}")
                
                for file in os.listdir(os.path.join(audio_dir, folder)):                    
                    
                    if (file.endswith(".wav")) and (not file.startswith('._')) and (not 'augmented' in file):
                        audio_features = self._extract_audio_features(audio_dir, folder, file)

                        # check if audio_features is a torch tensor
                        # if not isinstance(audio_features, torch.Tensor):
                        #     audio_features = torch.tensor(audio_features).float()
                    
                        if training_config['break_audio_into_chunks']:
                            # only use chunk files
                            if ('chunk' in file) and (not file.startswith('._')) and (file.endswith(".wav")):
                                if self.BINARY_CLASSIFICATION:
                                    logging.debug("Using binary classification")
                                    examples.append(
                                        {
                                            "file": os.path.join(audio_dir, folder, file),
                                            "audio": os.path.join(audio_dir, folder, file),
                                            "audio_features": audio_features,
                                            # from 300P to 300
                                            # "label": torch.tensor(int(label_file.loc[int(folder[:3])]["PHQ8_Binary"]))
                                            "label": int(label_file.loc[int(folder[:3])]["PHQ8_Binary"])
                                        }
                                    )
                                else:
                                    examples.append(
                                        {
                                            "file": os.path.join(audio_dir, folder, file),
                                            "audio": os.path.join(audio_dir, folder, file),
                                            "audio_features": audio_features,
                                            # from 300P to 300
                                            # "label": torch.tensor(place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True)),
                                            "label": place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True),
                                        }
                                    )
                        else:
                            if (file.endswith("PARTICIPANT_merged.wav")) and (not file.startswith('._')):                        
                                if self.BINARY_CLASSIFICATION:
                                    logging.debug("Using binary classification")
                                    examples.append(
                                        {
                                            "file": os.path.join(audio_dir, folder, file),
                                            "audio": os.path.join(audio_dir, folder, file),
                                            "audio_features": audio_features,
                                            # from 300P to 300
                                            # "label": torch.tensor(int(label_file.loc[int(folder[:3])]["PHQ8_Binary"]))
                                            "label": int(label_file.loc[int(folder[:3])]["PHQ8_Binary"])
                                        }
                                    )
                                else:
                                    examples.append(
                                        {
                                            "file": os.path.join(audio_dir, folder, file),
                                            "audio": os.path.join(audio_dir, folder, file),
                                            "audio_features": audio_features,
                                            # from 300P to 300
                                            # "label": torch.tensor(place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True)),
                                            "label": place_value_in_bin(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]), put_in_bin=True),
                                        }
                                    )
        
        # balance the data only if train
        if name == "train":
            examples = self._balance_data(examples)

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
                        examples.append(
                            {
                                "file": os.path.join(audio_dir, folder, file),
                                "audio": os.path.join(audio_dir, folder, file),
                                # from 300P to 300
                                "label": int(label_file.loc[int(folder[:3])]["PHQ8_Score"]),
                            }
                        )

        return examples


    # def __len__(self):
    #     return len(self.get_data(self.split))