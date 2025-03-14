#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

#python main.py --lr 0.001 --timepoints R1 R2 --data_path "$DATA_PATH"`
#python main.py --lr 0.001 --timepoints R1 R2 R3 R4 --data_path "$DATA_PATH"
#python main.py --lr 0.001 --timepoints R1 R2 R3 --data_path "$DATA_PATH"
python main.py --lr 0.0005 --timepoints R1 R2 R3 R4 R5 R6 --data_path "$DATA_PATH"
