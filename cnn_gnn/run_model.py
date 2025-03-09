import os
import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split
from model import CNN_GNN_Model
from helpers.feature_preparation import ImageDataProcessor
from helpers.utils import (
    build_weighted_graph, 
    custom_loss, 
    calcualte_metrics,
    save_best_model_predictions)
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*scatter\(reduce='max'\).*accelerated via the 'torch-scatter' package, but it was not found",
    category=UserWarning
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def train_model(data_dict, weight_matrix, edge_index, edge_weights, results_output_path, args):
    # -----------------------------------------------------------------
    # Prepare stratification labels using yield and irrigation_labels
    # -----------------------------------------------------------------
    kbins = KBinsDiscretizer(n_bins=args.n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbins.fit_transform(data_dict['yield'].reshape(-1, 1)).flatten().astype(int)
    stratify_labels = data_dict['irrigation_labels'] + "_" + y_binned.astype(str)
    stratify_encoded = LabelEncoder().fit_transform(stratify_labels)

    # -----------------------------------------------------------------
    # Train / Validation / Test Splits
    # -----------------------------------------------------------------
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

    # ---------------------------
    # Scale the image-like features using training data only
    # ---------------------------
    # Scale vegetation: shape (n_samples, channels, H, W)
    veg_np = data_dict['vegetation']
    n, c, h, w = veg_np.shape
    veg_scaled = np.empty_like(veg_np)

    for i in range(c):
        scaler = StandardScaler()
        # Fit scaler on the training subset for this channel
        train_channel_data = veg_np[train_idx, i, :, :].reshape(len(train_idx), -1)
        scaler.fit(train_channel_data)
        # Transform the entire channel using the training parameters
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

    # ---------------------------
    # Convert data to tensors
    # ---------------------------
    veg_all = torch.tensor(data_dict['vegetation'], dtype=torch.float32).to(DEVICE)
    cwsi_all = torch.tensor(data_dict['cwsi'], dtype=torch.float32).to(DEVICE)
    irrigation_all = torch.tensor(data_dict['irrigation'], dtype=torch.long).to(DEVICE)
    y_all = torch.tensor(data_dict['yield'], dtype=torch.float32).to(DEVICE)

    # ---------------------------
    # Initialize model, optimizer, scheduler, and loss function
    # ---------------------------
    model = CNN_GNN_Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience)
    # criterion = log_cosh_loss()

    best_model_path = os.path.join(results_output_path, 'best_model.pt')
    best_val_mse = float('inf')
    train_mse_list, val_mse_list, test_mse_list, train_r2_list, val_r2_list, test_r2_list = [], [], [], [], [], []

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass on all nodes (global graph)
        preds_all = model(veg_all, cwsi_all, irrigation_all, edge_index)

        # Calculate training and validation losses
        train_loss = custom_loss(preds_all[train_idx].squeeze(), y_all[train_idx], args.loss_method)
        val_loss = custom_loss(preds_all[val_idx].squeeze(), y_all[val_idx], args.loss_method)

        # Backpropagation using training loss
        train_loss.backward()
        optimizer.step()
        scheduler.step(val_loss)

        # Evaluation on test split
        model.eval()
        with torch.no_grad():
            # train
            train_preds = model(veg_all, cwsi_all, irrigation_all, edge_index)[train_idx].cpu().numpy()
            y_train = data_dict['yield'][train_idx]
            train_metrics = calcualte_metrics(y_train, train_preds)

            # val
            val_preds = model(veg_all, cwsi_all, irrigation_all, edge_index)[val_idx].cpu().numpy()
            y_val = data_dict['yield'][val_idx]
            val_metrics = calcualte_metrics(y_val, val_preds)

            # test
            test_preds = model(veg_all, cwsi_all, irrigation_all, edge_index)[test_idx].cpu().numpy()
            y_test = data_dict['yield'][test_idx]
            test_metrics = calcualte_metrics(y_test, test_preds)

            val_mse = val_metrics['mse']

            train_mse_list.append(train_metrics['mse'])
            val_mse_list.append(val_metrics['mse'])
            test_mse_list.append(test_metrics['mse'])
            train_r2_list.append(train_metrics['r2'])
            val_r2_list.append(val_metrics['r2'])
            test_r2_list.append(test_metrics['r2'])

        if epoch % 10 == 0 or epoch == args.epochs - 1:
          print(f"Epoch {epoch+1}/500 | "
                f"Train MSE: {train_metrics['mse']:.4f} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | "
                f"Val Loss: {val_loss.item():.4f} |"
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Test MSE: {test_metrics['mse']:.4f} | "
                f"Test R2: {test_metrics['r2']:.4f} | ")


        # Save best model
        if val_mse < best_val_mse:
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

    print("\nBest results:")
    print(f"Train - MSE:{best_train_mse:.4f}, MAE:{best_train_mae:.4f}, R2:{best_train_r2:.4f}")
    print(f"Val - MSE:{best_val_mse:.4f}, MAE:{best_val_mae:.4f}, R2:{best_val_r2:.4f}")
    print(f"Test - MSE:{best_test_mse:.4f}, MAE:{best_test_mae:.4f}, R2:{best_test_r2:.4f}")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        # Obtain predictions for all samples using the best model
        best_preds_all = model(veg_all, cwsi_all, irrigation_all, edge_index).cpu().numpy()

    save_best_model_predictions(data_dict, train_idx, val_idx, test_idx, 
                                best_preds_all, 
                                os.path.join(results_output_path,'predictions.csv'))


    #plot_metrics(train_mse_list, val_mse_list, test_mse_list, train_r2_list, val_r2_list, test_r2_list, best_epoch)
    
    
def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    processor = ImageDataProcessor(os.path.join(PROJECT_ROOT, args.data_path))

    for idx, timepoint in enumerate(['R1', 'R2', 'R3', 'R4', 'R5', 'R6']):
        print(f"\nProcessing timepoint: {timepoint}")

        # Load data once per timepoint
        data_dict = processor.load_and_process(timepoint=timepoint)

        # Only build the graph on the first iteration (R1)
        if idx == 0:
            sigma, weight_matrix = build_weighted_graph(
                data_dict['coordinates'], dist_scale=args.dist_scale
            )
            print("Sigma:", sigma)
            # display_weight_matrix(weight_matrix)
            # show_connected_graph(data_dict['coordinates'], sigma)
            edge_index = torch.from_numpy(np.vstack(weight_matrix.nonzero())).long().to(DEVICE)
            edge_weights = torch.from_numpy(weight_matrix[weight_matrix.nonzero()]).float().to(DEVICE)

        results_output_path = os.path.join(PROJECT_ROOT, args.model_type, 'results', timepoint)
        if not os.path.exists(results_output_path):
            os.makedirs(results_output_path)

        # Run stratified KFold training
        train_model(data_dict, weight_matrix, edge_index, edge_weights, results_output_path, args)
