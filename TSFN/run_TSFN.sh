#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

python main.py --lr 0.0009 --lr_patience 25 --timepoints R1 R2 R3 R4 R5 R6 --data_path "$DATA_PATH"
#python main.py --lr 0.01 --lr_patience 25 --timepoints R2 R3 R4 R5 R6  --data_path "$DATA_PATH"
#python main.py --lr 0.001 --lr_patience 25 --timepoints R1 R3 R4 R5 R6 --data_path "$DATA_PATH"
#python main.py --lr 0.001 --lr_patience 25 --timepoints R1 R2 R4 R5 R6 --data_path "$DATA_PATH"
#python main.py --lr 0.001 --lr_patience 25 --timepoints R1 R2 R3 R5 R6 --data_path "$DATA_PATH"
#python main.py --lr 0.001 --lr_patience 25 --timepoints R1 R2 R3 R4 R6 --data_path "$DATA_PATH"
#python main.py --lr 0.001 --lr_patience 25 --timepoints R1 R2 R3 R4 R5 --data_path "$DATA_PATH"
