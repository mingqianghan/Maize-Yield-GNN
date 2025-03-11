#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
# DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

# Activate Anaconda environment (local)
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate base

# Define the data path variable. (change this)
# local
DATA_PATH="C:/Users/mingq/OneDrive - Kansas State University/K-state Research/Publications/Maize Yield Prediction/Maize-Yield-GNN/dataset/stat_dataset.pkl"


for tp in R1 R2 R3 R4 R5 R6
    do
        python main.py --timepoint $tp --epochs 400 --data_path "$DATA_PATH"
done
