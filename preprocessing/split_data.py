import os
import shutil
import random
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

random.seed(42)

data_path = '/home/snag0027/daic_woz'

# create 3 folders: train, test and val
# os.mkdir(os.path.join(data_path, 'splits', 'train'))
# os.mkdir(os.path.join(data_path, 'splits', 'test'))
# os.mkdir(os.path.join(data_path, 'splits', 'val'))

# randomly split the other folders in data_path into the 3 folders
files = os.listdir(data_path)
# remove the splits folder
files.remove('splits')
files.remove('split_data.py')
# remove any strings that start with a dot
files = [x for x in files if not x.startswith('.')]
# remove any strings that are not a directory
files = [x for x in files if os.path.isdir(os.path.join(data_path, x))]
# only keep strings where the first 3 characters are numbers
files = [x for x in files if x[:3].isnumeric()]
# only keep the first 3 characters of each string
files = [int(x[:3]) for x in files]

train_files = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_files = pd.read_csv(os.path.join(data_path, 'test.csv'))
validation_files = pd.read_csv(os.path.join(data_path, 'validation.csv'))

train_files = train_files['Participant_ID'].tolist()
test_files = test_files['Participant_ID'].tolist()
validation_files = validation_files['Participant_ID'].tolist()

print(train_files)
print(validation_files)
print(test_files)

# copy each audio folder into the splits folder
for folder in files:
    print(type(train_files[0]))
    if folder in train_files:
        logging.info(f"Moving {folder} to train")
        shutil.move(os.path.join(data_path, f"{folder}_P"), os.path.join(data_path, 'splits', 'train', f"{folder}_P"))
    elif folder in test_files:
        logging.info(f"Moving {folder} to test")
        shutil.move(os.path.join(data_path, f"{folder}_P"), os.path.join(data_path, 'splits', 'test', f"{folder}_P"))
    elif folder in validation_files:
        logging.info(f"Moving {folder} to val")
        shutil.move(os.path.join(data_path, f"{folder}_P"), os.path.join(data_path, 'splits', 'val', f"{folder}_P"))
    else:
        print(f"Folder {folder} not in any of the splits")