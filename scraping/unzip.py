# unzip all files in a directory
import os
import zipfile

directory = '/Volumes/Files/monash/daic_woz/'

for filename in os.listdir(directory):
    if filename.endswith(".zip") and filename.split('.')[0] not in os.listdir(directory):
        print(os.path.join(directory, filename))
        try:
            with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                zip_ref.extractall(path=directory+filename.split('.')[0])
                print(f'Extracted {filename}')
        except:
            print(f'Failed to extract {filename}')
        continue
    else:
        continue