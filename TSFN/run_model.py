import os
import sys
import torch
import random
import numpy as np
import subprocess
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split 
from model import TSFN_Model
from helpers.feature_preparation import DataProcessor
from helpers.utils import (
    build_weighted_graph, 
    custom_loss, 
    calculate_metrics,
    get_git_revision_hash,
    save_best_model_predictions
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def train_model(data_dict, weight_matrix, edge_index, edge_weights, results_output_path, args):
    """
    Train the TSFN_Model on the provided data and record the best results.
    
    The training process includes:
        - Stratification of samples based on binned yield and irrigation labels.
        - Splitting data into training, validation, and test sets.
        - Scaling of vegetation and CWSI image-like features using training statistics.
        - Forward and backward passes of the TSFN_Model.
        - Model evaluation and checkpointing based on validation performance.
        - Saving predictions, attention weights, and experiment summary.
        - Optionally committing results to a Git repository.
    
    Parameters:
        data_dict (dict): Dictionary containing all input features and targets.
        weight_matrix (np.ndarray): Weight matrix for constructing the graph.
        edge_index (Tensor): Graph edge indices for GraphSAGE layers.
        edge_weights (Tensor): Graph edge weights.
        results_output_path (str): Directory path for saving outputs.
        args: Namespace with experiment hyperparameters and configuration.
    """

    # Prepare stratification labels using yield and irrigation info.
    kbins = KBinsDiscretizer(n_bins=args.n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbins.fit_transform(data_dict['yield'].reshape(-1, 1)).flatten().astype(int)
    stratify_labels = data_dict['irrigation_labels'] + "_" + y_binned.astype(str)
    stratify_encoded = LabelEncoder().fit_transform(stratify_labels)

    
    # Train / Validation / Test Splits
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

   # Scale the vegetation image-like features using training data only.
    veg_np = data_dict['vegetation']
    n, t, c, h, w = veg_np.shape  # n: samples, t: timepoints, c: channels, h: height, w: width
    veg_scaled = np.empty_like(veg_np)

    for i in range(c):
        scaler = StandardScaler()
        # Fit scaler on the training subset for this channel
        train_channel_data = veg_np[train_idx, :, i, :, :].reshape(len(train_idx) * t, -1)
        scaler.fit(train_channel_data)
        # Transform the entire channel using the training parameters
        channel_data = veg_np[:, :, i, :, :].reshape(n * t, -1)
        channel_data_scaled = scaler.transform(channel_data)
        veg_scaled[:, :, i, :, :] = channel_data_scaled.reshape(n, t, h, w)
    data_dict['vegetation'] = veg_scaled

     # Scale CWSI image-like features.
    cwsi_np = data_dict['cwsi']
    n, t, c, h, w = cwsi_np.shape  # c should be 1
    scaler = StandardScaler()
    # Fit scaler on the training subset.
    train_cwsi = cwsi_np[train_idx].reshape(len(train_idx) * t, -1)
    scaler.fit(train_cwsi)
    cwsi_np_scaled = scaler.transform(cwsi_np.reshape(n * t, -1)).reshape(n, t, c, h, w)
    data_dict['cwsi'] = cwsi_np_scaled

    
    # Convert all data to PyTorch tensors.
    veg_all = torch.tensor(data_dict['vegetation'], dtype=torch.float32).to(DEVICE)      # (N, T, 5, H, W)
    cwsi_all = torch.tensor(data_dict['cwsi'], dtype=torch.float32).to(DEVICE)           # (N, T, 1, H, W)
    irrigation_all = torch.tensor(data_dict['irrigation'], dtype=torch.long).to(DEVICE)  # (N, 1)
    y_all = torch.tensor(data_dict['yield'], dtype=torch.float32).to(DEVICE)             # (N,)

    # Initialize the model, optimizer, scheduler, and loss function.
    model = TSFN_Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience)

    
    # Define paths for saving the best model and attention weights.
    best_model_path = os.path.join(results_output_path, f'best_model_{args.seed}.pt')
    best_attn_path = os.path.join(results_output_path, f'attention_weights_{args.seed}.npy')
    best_val_mse = float('inf')
    # Lists to store metrics for each epoch.
    train_mse_list, val_mse_list, test_mse_list = [], [], []
    train_r2_list, val_r2_list, test_r2_list = [], [], []

    
    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass through the model. Input: full time-series.
        preds_all = model(veg_all, cwsi_all, irrigation_all, edge_index).squeeze() 

        # Compute loss on training data.
        train_loss = custom_loss(preds_all[train_idx], y_all[train_idx], args.loss_method)

        # Backward pass and optimization step.
        train_loss.backward()
        optimizer.step()

        # Evaluate model on training, validation, and test splits.
        model.eval()
        with torch.no_grad():
            preds_all_eval = model(veg_all, cwsi_all, irrigation_all, edge_index).squeeze()
            
            # Compute validation loss.
            val_loss = custom_loss(preds_all_eval[val_idx], y_all[val_idx], args.loss_method)
            
            # Compute predictions for each split.
            train_preds = preds_all_eval[train_idx].cpu().numpy()
            val_preds = preds_all_eval[val_idx].cpu().numpy()
            test_preds = preds_all_eval[test_idx].cpu().numpy()
        
            # Convert ground truth values for metric calculations.
            y_train_np = y_all[train_idx].cpu().numpy()
            y_val_np = y_all[val_idx].cpu().numpy()
            y_test_np = y_all[test_idx].cpu().numpy()
            
            # Calculate performance metrics.
            train_metrics = calculate_metrics(y_train_np, train_preds)
            val_metrics = calculate_metrics(y_val_np, val_preds)
            test_metrics = calculate_metrics(y_test_np, test_preds)

            # Store metrics.
            train_mse_list.append(train_metrics['mse'])
            val_mse_list.append(val_metrics['mse'])
            test_mse_list.append(test_metrics['mse'])
            train_r2_list.append(train_metrics['r2'])
            val_r2_list.append(val_metrics['r2'])
            test_r2_list.append(test_metrics['r2'])
        
        # Update learning rate scheduler.
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train MSE: {train_metrics['mse']:.4f} | Train Loss: {train_loss.item():.4f} | "
                  f"Val MSE: {val_metrics['mse']:.4f} | Val Loss: {val_loss.item():.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Test MSE: {test_metrics['mse']:.4f} | Test R2: {test_metrics['r2']:.4f}")

        # Save the best model based on validation MSE.
        if val_metrics['mse']  < best_val_mse:
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

    # Print progress every 10 epochs or at the final epoch.
    print("\nBest results:")
    print(f"Train - MSE: {best_train_mse:.4f}, MAE: {best_train_mae:.4f}, R2: {best_train_r2:.4f}")
    print(f"Val   - MSE: {best_val_mse:.4f}, MAE: {best_val_mae:.4f}, R2: {best_val_r2:.4f}")
    print(f"Test  - MSE: {best_test_mse:.4f}, MAE: {best_test_mae:.4f}, R2: {best_test_r2:.4f}")

    # Final predictions and attention weights.
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        best_preds_all = model(veg_all, cwsi_all, irrigation_all, edge_index).squeeze().cpu().numpy()
        attn_weights = model.get_attention_weights(veg_all[test_idx], cwsi_all[test_idx])

    # Save attention weights and predictions.
    np.save(best_attn_path, attn_weights.cpu().detach().numpy())
    predictions_file = os.path.join(results_output_path, f'predictions_{args.seed}.csv')
    save_best_model_predictions(data_dict, train_idx, val_idx, test_idx, best_preds_all, predictions_file)

     # Save experiment summary and record Git information.
    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)
    summary_txt = os.path.join(results_output_path, f'summary_{args.seed}.txt')
    with open(summary_txt, 'w') as f:
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n")
        f.write("Best result at epoch {:d}, \n".format(best_epoch))
        f.write("TRAIN Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_train_mse, best_train_mae, best_train_r2))
        f.write("VAL   Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_val_mse, best_val_mae, best_val_r2))
        f.write("TEST  Metrics | MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n".format(best_test_mse, best_test_mae, best_test_r2))

    try:
        files_to_commit = [summary_txt, best_model_path, best_attn_path, predictions_file]
        subprocess.check_call(["git", "add"] + files_to_commit)
        commit_message = "Record final results"
        subprocess.check_call(["git", "commit", "-m", commit_message])
        subprocess.check_call(["git", "push"])
        print("Results committed and pushed to GitHub.\n")
    except subprocess.CalledProcessError as e:
        print("Error during Git operations:", e)


def run(args):
    """
    Main function to run the experiment.
    
    It sets the random seed, loads and processes the data,
    builds the weighted graph, and launches the training process.
    
    Parameters:
        args: Namespace containing experiment configuration and hyperparameters.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and process the dataset.
    processor = DataProcessor(args.data_path)
    data_dict = processor.load_and_process_all_timepoints(args.timepoints)
    
    # Build a weighted graph using spatial coordinates.
    sigma, weight_matrix = build_weighted_graph(
        data_dict['coordinates'], dist_scale=args.dist_scale
    )
    print("Sigma:", sigma)
    edge_index = torch.from_numpy(np.vstack(weight_matrix.nonzero())).long().to(DEVICE)
    edge_weights = torch.from_numpy(weight_matrix[weight_matrix.nonzero()]).float().to(DEVICE)
    
    # Create directory for saving results.
    results_output_path = os.path.join(PROJECT_ROOT, args.model_type, 'results', '_'.join(args.timepoints))
    if not os.path.exists(results_output_path):
        os.makedirs(results_output_path)

    # Create directory for saving results.
    train_model(data_dict, weight_matrix, edge_index, edge_weights, results_output_path, args)