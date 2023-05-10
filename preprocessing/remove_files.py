# recursively read files in "/home/snag0027/daic_woz/old_splits"
# delete any .wav file that contains the word "chunk"

# recursively read folders and files in "/home/snag0027/daic_woz/old_splits"
import os


path = "/home/snag0027/daic_woz/old_splits"
for split in os.listdir(path):
    for folder in os.listdir(os.path.join(path, split)):
        for file in os.listdir(os.path.join(path, split, folder)):
            if file.endswith(".wav"):
                if "chunk" in file:
                    print(f"Removing {file}")
                    os.remove(os.path.join(path, split, folder, file))