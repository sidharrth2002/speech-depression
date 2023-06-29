#!/bin/bash

#SBATCH --job-name=astmultistream
#SBATCH --time=40:00:00
#SBATCH --mem=16384
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu.medium

module load python
pip install transformers datasets evaluate librosa torchmetrics opensmile scikit-learn nlpaug

python /home/snag0027/speech-depression/cluster/train_1d_conv.py