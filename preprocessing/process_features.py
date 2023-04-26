import os
from feature_names import *
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

with open('/home/snag0027/cluster/sample_features_is09.txt', 'r') as f:
    sample_features = f.read()
    num_features_is09 = 0
    for line in sample_features.splitlines():
        if line.startswith('@attribute'):
            num_features_is09 += 1

with open('/home/snag0027/cluster/sample_features_egemaps.txt', 'r') as f:
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
        for line in f:
            if line.startswith("'unknown'"):
                data = line
                break
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

datafolder = '/home/snag0027/daic_woz'

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
                print(is09[audio_name])
            if file.endswith("eGeMAPSv01b.txt"):
                egemaps[audio_name] = process_features(os.path.join(datafolder, folder, file))
                print(egemaps[audio_name])

# create a dataframe from the dictionary

df = pd.DataFrame.from_dict(is09, columns=is09_feature_names)
df.to_csv('features_is09.csv')

egemaps = pd.DataFrame.from_dict(egemaps, columns=egemaps_feature_names)
egemaps.to_csv('features_egemaps.csv')

# concatenate the two dataframes based on the index
df = pd.concat([df, egemaps], axis=1)

df.to_csv('/home/snag0027/daic_woz/features.csv')