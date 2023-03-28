# Lint as: python3
"""DAIC-WOZ Dataset."""
"""Main file for DAIC-WOZ dataset"""

'''
Custom huggingface dataloader to return audio and labels
'''
import json
import os
import datasets
import pandas as pd

class DaicWozConfig(datasets.BuilderConfig):
    """
    Builder config for DAIC-WOZ dataset
    """

    def __init__(self, **kwargs):
        super(DaicWozConfig, self).__init__(
            version=datasets.Version("0.0.1", ""), **kwargs)


class DaicWozDataset(datasets.GeneratorBasedBuilder):
    '''
    Return only the audio and labels
    '''
    BUILDER_CONFIGS = [
        DaicWozConfig(
            name="daic_woz",
            description="DAIC-WOZ dataset",
        ),
    ]


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

        for folder in os.listdir(audio_dir):
            # check if folder is a directory and that the first 3 characters are numeric
            if os.path.isdir(os.path.join(audio_dir, folder)) and folder[:3].isnumeric():
                for file in os.listdir(os.path.join(audio_dir, folder)):
                    if file.endswith("PARTICIPANT_merged.wav") and not file.startswith('._'):
                        print(int(label_file.loc[int(folder[:3])]["PHQ8_Score"]))
                        examples.append(
                            {
                                "file": os.path.join(audio_dir, folder, file),
                                "audio": os.path.join(audio_dir, folder, file),
                                # from 300P to 300
                                "label": int(label_file.loc[int(folder[:3])]["PHQ8_Score"]),
                            }
                        )

        with open("data.json", "w") as f:
            json.dump(examples, f, indent=4)

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

        # load labels file
        # if name == "train":
        #     label_file = os.path.join(audio_dir, "train.csv")
        # elif name == "validation":
        #     label_file = os.path.join(audio_dir, "validation.csv")
        # else:
        #     label_file = os.path.join(audio_dir, "test.csv")

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


    def __len__(self):
        return len(self.get_data(self.split))
    