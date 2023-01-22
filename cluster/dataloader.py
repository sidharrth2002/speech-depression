'''
Custom huggingface dataloader to return audio and labels
'''
import os
import datasets
import pandas as pd


class DaicWozDataset(datasets.GeneratorBasedBuilder):
    '''
    Return only the audio and labels
    '''

    def _info(self):
        return datasets.DatasetInfo(
            description="DAIC-WOZ dataset",
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "label"),
        )

    def _split_generators(self, dl_manager):
        # already downloaded to a specified directory
        data_dir = '/Volumes/Files/monash/daic_woz/'
        if self.config.name == 'clean':
            train_splits = [
                datasets.SplitGenerator(
                    name="train", gen_kwards={"files": data_dir, "name": "train"}
                ),
                datasets.SplitGenerator(
                    name="validation", gen_kwards={"files": data_dir, "name": "validation"}
                ),
                datasets.SplitGenerator(
                    name="test", gen_kwards={"files": data_dir, "name": "test"}
                ),
            ]
        return train_splits

    def _generate_examples(self, files, name):
        key = 0
        examples = []

        audio_dir = "/Volumes/Files/monash/daic_woz/"

        if not os.path.exists(audio_dir):
            raise FileNotFoundError(
                f"{audio_dir} does not exist. Make sure you insert the correct path to the audio directory."
            )

        # load labels file
        if name == "train":
            label_file = os.path.join(audio_dir, "train.csv")
        elif name == "validation":
            label_file = os.path.join(audio_dir, "validation.csv")
        else:
            label_file = os.path.join(audio_dir, "test.csv")

        label_file = pd.read_csv(label_file)
        # set Participant_ID as index
        label_file = label_file.set_index("Participant_ID")

        for folder in os.listdir(audio_dir):
            # check if folder is a directory and that the first 3 characters are numeric
            if os.path.isdir(os.path.join(audio_dir, folder)) and folder[:3].isnumeric():
                for file in os.listdir(os.path.join(audio_dir, folder)):
                    if file.endswith("PARTICIPANT_merged.wav"):
                        examples.append(
                            {
                                "file": os.path.join(audio_dir, folder, file),
                                # from 300P to 300
                                "label": label_file.loc[int(folder[:3])]["PHQ8_Score"],
                            }
                        )

        for example in examples:
            yield key, {**example}
            key += 1
        examples = []
