import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import subprocess
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def get_git_revision_hash():
    """
    Retrieve the current Git commit hash.

    Returns:
        str: The Git revision hash.
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance between two GPS coordinates.

    Parameters:
        coord1 (array-like): [lat1, lon1] in degrees.
        coord2 (array-like): [lat2, lon2] in degrees.

    Returns:
        float: Distance between the two coordinates in meters.
    """
    # Convert degrees to radians.
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    # Compute differences.
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula.
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # Earth's radius in meters.
    return R * c


def build_weighted_graph(coords_all, dist_scale, threshold=0.001):
    """
    Build a weight matrix for a graph using a Gaussian kernel based on GPS coordinates.

    Parameters:
        coords_all (ndarray): Array of shape (n_samples, 2) containing [latitude, longitude].
        dist_scale (float): Multiplier applied to the median distance to compute sigma.
        threshold (float, optional): Minimum weight for an edge to be retained. Defaults to 0.001.

    Returns:
        tuple: (sigma, weight_matrix) where sigma is the computed bandwidth and weight_matrix is an ndarray of shape (n_samples, n_samples).
    """
    # Compute full pairwise distances using the Haversine metric.
    pairwise_dists = squareform(pdist(coords_all, metric=haversine_distance))
    median_distance = np.median(pairwise_dists)
    sigma = dist_scale * median_distance

    # Compute the weight matrix using a Gaussian kernel.
    weight_matrix = np.exp(- (pairwise_dists ** 2) / (2 * sigma ** 2))
    # Filter out weak connections.
    weight_matrix[weight_matrix < threshold] = 0
    # Remove self-loops.
    np.fill_diagonal(weight_matrix, 0)
    return sigma, weight_matrix


def show_connected_graph(coords_all, sigma, threshold=0.001):
    """
    Build and display a connected graph from GPS coordinates.

    Parameters:
        coords_all (ndarray): Array of shape (n_samples, 2) with each row as [latitude, longitude].
        sigma (float): Bandwidth parameter for the Gaussian kernel (in meters).
        threshold (float, optional): Minimum weight for an edge to be included in the graph. Defaults to 0.001.
    """
    # Build weight matrix using provided sigma.
    _, weight_matrix = build_weighted_graph(coords_all, sigma, threshold)

    # Create a NetworkX graph.
    G = nx.Graph()
    n_samples = coords_all.shape[0]

    # Add nodes with position attributes (longitude as x, latitude as y).
    for i in range(n_samples):
        lat, lon = coords_all[i]
        G.add_node(i, pos=(lon, lat))

    # Add edges for nonzero weights.
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if weight_matrix[i, j] > 0:
                G.add_edge(i, j, weight=weight_matrix[i, j])

    # Plot the graph.
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.title("Connected Graph from GPS Coordinates")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def plot_metrics(train_mse_list, val_mse_list, test_mse_list,
                 train_r2_list, val_r2_list, test_r2_list, best_epoch=None):
    """
    Plot training, validation, and test metrics (MSE and R2) over epochs.

    Parameters:
        train_mse_list (list): MSE values for training data per epoch.
        val_mse_list (list): MSE values for validation data per epoch.
        test_mse_list (list): MSE values for test data per epoch.
        train_r2_list (list): R2 values for training data per epoch.
        val_r2_list (list): R2 values for validation data per epoch.
        test_r2_list (list): R2 values for test data per epoch.
        best_epoch (int, optional): Epoch index of the best model (for visual reference). Defaults to None.
    """
    epochs = range(1, len(train_mse_list) + 1)

    plt.figure(figsize=(14, 6))

    # Plot MSE metrics.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_mse_list, label='Train MSE')
    plt.plot(epochs, val_mse_list, label='Validation MSE')
    plt.plot(epochs, test_mse_list, label='Test MSE')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE vs. Epoch')
    plt.legend(loc='center right')

    # Plot R2 metrics.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_r2_list, label='Train R2')
    plt.plot(epochs, val_r2_list, label='Validation R2')
    plt.plot(epochs, test_r2_list, label='Test R2')
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.title('R2 vs. Epoch')
    plt.legend(loc='center right')

    plt.tight_layout()
    plt.show()


def calculate_correlation_coefficient_all_timepoints(data, show_plot=False):
    """
    Compute Pearson correlation coefficients between each feature and the target 'Yield'
    for all six timepoints.

    Optionally, plots a regression plot for each feature.

    Parameters:
        data (pandas.DataFrame): The dataset containing a 'timepoint' column and 'Yield'.
        show_plot (bool, optional): If True, displays regression plots for each feature. Defaults to False.

    Returns:
        dict: A dictionary of correlation coefficients keyed by timepoint and feature.
    """
    corr_dict = {}

    for tp in range(1, 7):
        stage = f"R{tp}"
        df = data[data['timepoint'] == stage].dropna(subset=['Yield'])
        features = df.columns[4:]  # Select features from the 5th column onward.
        target = 'Yield'

        corr_dict[stage] = {}
        for feature in features:
            # Compute Pearson correlation.
            corr, _ = pearsonr(df[feature], df[target])
            corr_dict[stage][feature] = corr

        if show_plot:
            num_features = len(features)
            num_stat = 5   # For example, mean, median, q1, q3, sum.
            rows = int(np.ceil(num_features / num_stat))
            cols = num_stat

            plt.rcParams['font.family'] = 'Times New Roman'
            fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
            axes = axes.flatten()  # Flatten to iterate easily.

            for i, feature in enumerate(features):
                ax = axes[i]
                sns.regplot(x=df[feature], y=df[target], 
                            scatter_kws={"alpha": 0.5, "s": 18},
                            line_kws={"color": "red"}, ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(f'{feature} r = {corr_dict[stage][feature]:.2f}', fontsize=10)

            fig.suptitle(f'Pearson Correlation Coefficients for {stage}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()

    return corr_dict


def select_features_with_high_correlation(df, target='yield', threshold=0.5):
    """
    Select features from a DataFrame that have an absolute Pearson correlation with the target
    above a given threshold.

    Parameters:
        df (pandas.DataFrame): DataFrame containing features and the target variable.
        target (str, optional): The target column name. Defaults to 'yield'.
        threshold (float, optional): The correlation threshold. Defaults to 0.5.

    Returns:
        list: List of feature names with correlation above the threshold.
    """
    exclude_cols = ['plot_id', 'irrigation_labels', 'irrigation']
    numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)
    df_numeric = df[numeric_cols]

    corr_matrix = df_numeric.corr(method='pearson')
    target_corr = corr_matrix[target].drop(target)
    selected_features = target_corr[abs(target_corr) > threshold].index.tolist()
    
    return selected_features


def save_best_model_predictions(data_dict, train_idx, val_idx, test_idx, best_preds_all, filepath):
    """
    Save best model predictions along with irrigation treatment information for train, validation,
    and test splits to a CSV file.

    Parameters:
        data_dict (dict): Dictionary containing data (including 'irrigation_labels' and 'yield').
        train_idx (array-like): Indices for the training split.
        val_idx (array-like): Indices for the validation split.
        test_idx (array-like): Indices for the test split.
        best_preds_all (np.array): Predictions for all samples from the best model.
        filepath (str): File path for saving the CSV.

    Returns:
        pandas.DataFrame: Combined DataFrame with all splits.
    """
    # Create DataFrame for training split.
    train_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][train_idx],
        'irrigation_trt': data_dict['irrigation_labels'][train_idx],
        'yield_true': data_dict['yield'][train_idx],
        'yield_pred': best_preds_all[train_idx].squeeze()
    })
    train_df['split'] = 'train'

    # Create DataFrame for validation split.
    val_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][val_idx],
        'irrigation_trt': data_dict['irrigation_labels'][val_idx],
        'yield_true': data_dict['yield'][val_idx],
        'yield_pred': best_preds_all[val_idx].squeeze()
    })
    val_df['split'] = 'validation'

    # Create DataFrame for test split.
    test_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][test_idx],
        'irrigation_trt': data_dict['irrigation_labels'][test_idx],
        'yield_true': data_dict['yield'][test_idx],
        'yield_pred': best_preds_all[test_idx].squeeze()
    })
    test_df['split'] = 'test'

    # Combine splits and save to CSV.
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df.to_csv(filepath, index=False)

    return all_df


def calculate_metrics(y_test, test_preds):
    """
    Compute regression metrics between true and predicted values.

    Parameters:
        y_test (array-like): Ground truth target values.
        test_preds (array-like): Predicted target values.

    Returns:
        dict: Dictionary with mean squared error (mse), mean absolute error (mae),
              R2 score (r2), and Pearson correlation (corr).
    """
    mse = mean_squared_error(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    corr = np.corrcoef(np.ravel(y_test), np.ravel(test_preds))[0, 1]
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}


def custom_loss(y_pred, y_true, loss_method='logcosh', huber_beta=1.0):
    """
    Compute the loss between predictions and true values using the specified loss method.

    Supported loss methods:
        - 'logcosh': Log-cosh loss.
        - 'mse': Mean squared error.
        - 'mae': Mean absolute error.
        - 'huber': Huber loss (requires huber_beta parameter).

    Parameters:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.
        loss_method (str, optional): Loss method to use. Defaults to 'logcosh'.
        huber_beta (float, optional): Beta parameter for Huber loss. Defaults to 1.0.

    Returns:
        torch.Tensor: Computed loss value.

    Raises:
        ValueError: If an unsupported loss method is specified.
    """
    loss_method = loss_method.lower()

    if loss_method == 'logcosh':
        diff = y_pred - y_true
        constant = torch.log(torch.tensor(2.0, device=y_pred.device))
        loss = torch.mean(diff + F.softplus(-2 * diff) - constant)
    elif loss_method == 'mse':
        loss = torch.mean((y_pred - y_true) ** 2)
    elif loss_method == 'mae':
        loss = torch.mean(torch.abs(y_pred - y_true))
    elif loss_method == 'huber':
        loss = F.smooth_l1_loss(y_pred, y_true, beta=huber_beta)
    else:
        raise ValueError("Unsupported loss_method. Supported methods are: 'logcosh', 'mse', 'mae', 'huber'.")
    
    return loss
