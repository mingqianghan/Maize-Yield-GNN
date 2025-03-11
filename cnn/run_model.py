import os
import sys
import torch
import random
import numpy as np
import subprocess
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split

from model import CNN_Model  # <-- your CNN-only model
from helpers.feature_preparation import DataProcessor
from helpers.utils import (
    custom_loss, 
    calculate_metrics,
    get_git_revision_hash,
    save_best_model_predictions
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def train_model(data_dict, results_output_path, args):
    """
    Trains a CNN model using vegetation, CWSI, and irrigation data.
    """

    # --------------------------------------------------
    # 1) Stratify & train/val/test split
    # --------------------------------------------------
    kbins = KBinsDiscretizer(n_bins=args.n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbins.fit_transform(data_dict['yield'].reshape(-1, 1)).flatten().astype(int)

    # Combine irrigation label with yield bin
    stratify_labels = data_dict['irrigation_labels'] + "_" + y_binned.astype(str)
    stratify_encoded = LabelEncoder().fit_transform(stratify_labels)

    indices = np.arange(len(stratify_encoded))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=stratify_encoded,
        random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size,
        stratify=stratify_encoded[train_idx],
        random_state=args.seed
    )

    # --------------------------------------------------
    # 2) Scale vegetation & CWSI channels using only training data
    # --------------------------------------------------
    veg_np = data_dict['vegetation']   # shape: (n_samples, num_channels, H, W)
    n, c, h, w = veg_np.shape
    veg_scaled = np.empty_like(veg_np)

    for i in range(c):
        scaler = StandardScaler()
        # Fit only on training subset for channel i
        train_channel_data = veg_np[train_idx, i, :, :].reshape(len(train_idx), -1)
        scaler.fit(train_channel_data)
        # Transform entire channel with that scaler
        channel_data = veg_np[:, i, :, :].reshape(n, -1)
        channel_data_scaled = scaler.transform(channel_data)
        veg_scaled[:, i, :, :] = channel_data_scaled.reshape(n, h, w)
    data_dict['vegetation'] = veg_scaled

    # Scale CWSI: shape (n_samples, 1, H, W)
    cwsi_np = data_dict['cwsi']
    n, c, h, w = cwsi_np.shape  # c should be 1
    scaler = StandardScaler()
    # Fit on the training data
    train_cwsi = cwsi_np[train_idx].reshape(len(train_idx), -1)
    scaler.fit(train_cwsi)
    cwsi_np_scaled = scaler.transform(cwsi_np.reshape(n, -1)).reshape(n, c, h, w)
    data_dict['cwsi'] = cwsi_np_scaled

    # --------------------------------------------------
    # 3) Convert arrays to tensors & create train, val, test splits
    # --------------------------------------------------
    veg_all = torch.tensor(data_dict['vegetation'], dtype=torch.float32)
    cwsi_all = torch.tensor(data_dict['cwsi'], dtype=torch.float32)
    irrigation_all = torch.tensor(data_dict['irrigation'], dtype=torch.long)
    y_all = torch.tensor(data_dict['yield'], dtype=torch.float32)

    # Move everything to GPU if available
    veg_all = veg_all.to(DEVICE)
    cwsi_all = cwsi_all.to(DEVICE)
    irrigation_all = irrigation_all.to(DEVICE)
    y_all = y_all.to(DEVICE)

    # Subset for TRAIN
    veg_train = veg_all[train_idx]
    cwsi_train = cwsi_all[train_idx]
    irrigation_train = irrigation_all[train_idx]
    y_train = y_all[train_idx]

    # Subset for VAL
    veg_val = veg_all[val_idx]
    cwsi_val = cwsi_all[val_idx]
    irrigation_val = irrigation_all[val_idx]
    y_val = y_all[val_idx]

    # Subset for TEST
    veg_test = veg_all[test_idx]
    cwsi_test = cwsi_all[test_idx]
    irrigation_test = irrigation_all[test_idx]
    y_test = y_all[test_idx]

    # --------------------------------------------------
    # 4) Initialize model, optimizer, scheduler
    # --------------------------------------------------
    model = CNN_Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_patience
    )

    best_val_mse = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(results_output_path, f'best_model_{args.seed}.pt')
    train_mse_list, val_mse_list, test_mse_list, train_r2_list, val_r2_list, test_r2_list = [], [], [], [], [], []

    # --------------------------------------------------
    # 5) Training loop
    # --------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # --- FORWARD + BACKPROP ON TRAIN SUBSET ---
        train_preds = model(veg_train, cwsi_train, irrigation_train).squeeze()
        train_loss = custom_loss(train_preds, y_train, args.loss_method)
        train_loss.backward()
        optimizer.step()

        # --- EVALUATION ON VAL & TEST ---
        model.eval()
        with torch.no_grad():
            # Validation forward pass
            val_preds = model(veg_val, cwsi_val, irrigation_val).squeeze()
            val_loss = custom_loss(val_preds, y_val, args.loss_method)

            # Learning rate schedule step
            scheduler.step(val_loss)

            # Test forward pass (optional to do every epoch)
            test_preds = model(veg_test, cwsi_test, irrigation_test).squeeze()

            # --- Calculate metrics ---
            train_metrics = calculate_metrics(y_train.cpu().numpy(), train_preds.cpu().numpy())
            val_metrics = calculate_metrics(y_val.cpu().numpy(), val_preds.cpu().numpy())
            test_metrics = calculate_metrics(y_test.cpu().numpy(), test_preds.cpu().numpy())
            
            train_mse_list.append(train_metrics['mse'])
            val_mse_list.append(val_metrics['mse'])
            test_mse_list.append(test_metrics['mse'])
            train_r2_list.append(train_metrics['r2'])
            val_r2_list.append(val_metrics['r2'])
            test_r2_list.append(test_metrics['r2'])


        # --- Print progress every 10 epochs or last epoch ---
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train MSE: {train_metrics['mse']:.4f} | "
                  f"Train Loss: {train_loss.item():.4f} | "
                  f"Val MSE: {val_metrics['mse']:.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Test MSE: {test_metrics['mse']:.4f} | "
                  f"Test R2: {test_metrics['r2']:.4f}")

        # --- Save best model based on validation MSE ---
        if val_metrics['mse'] < best_val_mse:
            best_train_mse = train_metrics['mse']
            best_train_mae = train_metrics['mae']
            best_train_r2 = train_metrics['r2']

            best_val_mse = val_metrics['mse']
            best_val_mae = val_metrics['mae']
            best_val_r2 = val_metrics['r2']

            best_test_mse = test_metrics['mse']
            best_test_mae = test_metrics['mae']
            best_test_r2 = test_metrics['r2']

            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

    # --------------------------------------------------
    # 6) Final results and predictions using best model
    # --------------------------------------------------
    print(f"\nBest epoch: {best_epoch+1}")
    print(f"Train - MSE:{best_train_mse:.4f}, MAE:{best_train_mae:.4f}, R2:{best_train_r2:.4f}")
    print(f"Val - MSE:{best_val_mse:.4f}, MAE:{best_val_mae:.4f}, R2:{best_val_r2:.4f}")
    print(f"Test - MSE:{best_test_mse:.4f}, MAE:{best_test_mae:.4f}, R2:{best_test_r2:.4f}")
    
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds_all = model(veg_all, cwsi_all, irrigation_all).cpu().numpy()
    
    # Save predictions, do git commit/push, etc.
    predictions_file = os.path.join(results_output_path, f'predictions_{args.seed}.csv')
    save_best_model_predictions(data_dict, train_idx, val_idx, test_idx, preds_all, predictions_file)

    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)

    summary_txt = os.path.join(results_output_path, f'summary_{args.seed}.txt')
    with open(summary_txt, 'w') as f:
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n")
        f.write(f"Best result at epoch {best_epoch}\n")
        f.write("TRAIN Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_train_mse, best_train_mae, best_train_r2))
        f.write("VAL Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_val_mse, best_val_mae, best_val_r2))
        f.write("TEST  Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_test_mse, best_test_mae, best_test_r2))

    try:
        # Optional: commit & push results to Git, if desired
        files_to_commit = [summary_txt, best_model_path, predictions_file]
        subprocess.check_call(["git", "add"] + files_to_commit)
        commit_message = "Record final results"
        subprocess.check_call(["git", "commit", "-m", commit_message])
        subprocess.check_call(["git", "push"])
        print("Results committed and pushed to GitHub.\n")
    except subprocess.CalledProcessError as e:
        print("Error during Git operations:", e)

def run(args):
    """
    Main entry point to process data and train the CNN model.
    """
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\nProcessing timepoint: {args.timepoint}")
    processor = DataProcessor(args.data_path)
    data_dict = processor.load_and_process(args.timepoint)

    results_output_path = os.path.join(PROJECT_ROOT, args.model_type, 'results', args.timepoint)
    if not os.path.exists(results_output_path):
        os.makedirs(results_output_path)

    # Train CNN model
    train_model(data_dict, results_output_path, args)
