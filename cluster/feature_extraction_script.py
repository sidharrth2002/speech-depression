import os
import subprocess
import logging

folder_path = '..'

num_files = 0
for folder in os.listdir(folder_path):
    for file in os.listdir(os.path.join(folder_path, folder)):
        if file.endswith("PARTICIPANT_merged.wav"):
            num_files += 1

logging.info(f"Found {num_files} files.")
logging.info("Starting feature extraction...")

for folder in os.listdir(folder_path):
    for file in os.listdir(os.path.join(folder_path, folder)):
        if file.endswith("PARTICIPANT_merged.wav"):
            # get the participant ID
            participant_id = file.split("_")[0]
            logging.info(f"Extracting IS09 features for {file}...")
            extraction_command = f"SMILExtract -C config/IS09_emotion.conf -I {os.path.join(folder_path, folder, file)} -O {os.path.join(folder_path, folder, participant_id + 'IS09.csv')}"
            subprocess.call(extraction_command, shell=True)
            logging.info(f"Extracting eGeMAPSv02 features for {file}...")
            extraction_command = f"SMILExtract -C config/eGeMAPSv02a.conf -I {os.path.join(folder_path, folder, file)} -O {os.path.join(folder_path, folder, participant_id + 'eGeMAPSv02.csv')}"
            subprocess.call(extraction_command, shell=True)
