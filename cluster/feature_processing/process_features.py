import os
from feature_processing.feature_names import *
import logging

logging.basicConfig(level=logging.INFO)

with open('sample_features_is09.txt', 'r') as f:
    sample_features = f.read()
    num_features_is09 = 0
    for line in sample_features.splitlines():
        if line.startswith('@attribute'):
            num_features_is09 += 1

print(f"Number of is09 features: {num_features_is09}")

with open('/home/snag0027/speech-depression/cluster/feature_processing/sample_features_egemaps.txt', 'r') as f:
    sample_features = f.read()
    num_features_egemaps = 0
    for line in sample_features.splitlines():
        if line.startswith('@attribute'):
            num_features_egemaps += 1

logging.info(f"Number of egemaps features: {num_features_egemaps}")
logging.info(f"Total number of features: {num_features_is09 + num_features_egemaps}")

def process_features(txt_file):
    with open(txt_file, 'r') as f:
        # find line that starts with @data
        # store data in the next line
        data = ''
        for line in f:
            if line.startswith("'unknown'"):
                data = line
                break
        if data == '':
            logging.info(f"Could not find data in {txt_file}")
            return [0] * (num_features_is09 + num_features_egemaps)
        # split data by comma
        data = data.split(',')
        # remove first and last element
        data = data[1:-1]
        # convert to float
        data = [float(x) for x in data]
        return data

def get_feature_names(attr_string):
    is09 = attr_string.split('@attribute')
    is09 = [x.strip() for x in is09]
    # remove \n from the end of each attribute
    is09 = [x.split(' ')[0] for x in is09]
    # remove any empty strings
    is09 = [x for x in is09 if x]

if __name__ == '__main__':
    datafolder = './daic_woz'

    is09 = {}
    egemaps = {}

    is09_feature_names = get_feature_names(is09_features)
    egemaps_feature_names = get_feature_names(egemaps_features)

    for folder in os.listdir(datafolder):
        if os.path.isdir(os.path.join(datafolder, folder)) and folder[:3].isnumeric():
            audio_name = folder[:3]
            for file in os.listdir(os.path.join(datafolder, folder)):
                if file.endswith("IS09.txt"):
                    is09[audio_name] = process_features(os.path.join(datafolder, folder, file))
                if file.endswith("eGeMAPSv01b.txt"):
                    egemaps[audio_name] = process_features(os.path.join(datafolder, folder, file))

    # create a dataframe from the dictionary
    import pandas as pd

    df = pd.DataFrame.from_dict(is09, orient='index', columns=is09_feature_names)
    print(df.head())