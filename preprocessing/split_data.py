import os
import shutil
import random

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

# split in 80-10-10
train_files = random.sample(files, int(len(files) * 0.8))
test_files = random.sample([x for x in files if x not in train_files], int(len(files) * 0.1))
val_files = [x for x in files if x not in train_files and x not in test_files]

# copy the folders to the train, test and val folders
for file in train_files:
    print(file)
    shutil.copytree(os.path.join(data_path, file), os.path.join(data_path, 'splits', 'train', file))
    
for file in test_files:
    print(file)
    shutil.copytree(os.path.join(data_path, file), os.path.join(data_path, 'splits', 'test', file))
    
for file in val_files:
    print(file)
    shutil.copytree(os.path.join(data_path, file), os.path.join(data_path, 'splits', 'val', file))
