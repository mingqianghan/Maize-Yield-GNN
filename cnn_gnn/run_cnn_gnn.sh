#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

python main.py --lr 0.001 --timepoint 'R1' --data_path "$DATA_PATH"
python main.py --lr 0.001 --timepoint 'R2' --data_path "$DATA_PATH"
python main.py --lr 0.001 --timepoint 'R3' --data_path "$DATA_PATH"
python main.py --lr 0.001 --timepoint 'R4' --data_path "$DATA_PATH"
python main.py --lr 0.001 --timepoint 'R5' --data_path "$DATA_PATH"
python main.py --lr 0.01 --timepoint 'R6' --data_path "$DATA_PATH"