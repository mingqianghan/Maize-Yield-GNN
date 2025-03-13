#!/bin/bash
# Windows Aanconda

# Define the data path variable. (change this)
# google colab
DATA_PATH="/content/drive/MyDrive/Colab Notebooks/Maize Yield Prediction/dataset/image_dataset.pkl"

python main.py --data_path "$DATA_PATH"
