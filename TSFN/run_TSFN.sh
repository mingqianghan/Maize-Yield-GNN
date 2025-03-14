#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

#python main.py --lr 0.001 --timepoints R1 R2 --data_path "$DATA_PATH"
python main.py --lr 0.01 --timepoints R1 R2 --loss_method mse --data_path "$DATA_PATH"
