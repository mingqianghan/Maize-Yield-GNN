import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance between two GPS coordinates.
    Inputs:
        coord1: array-like with [lat1, lon1] in degrees
        coord2: array-like with [lat2, lon2] in degrees
    Returns:
        Distance in meters.
    """
    # Convert degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # Earth's radius in meters
    return R * c

def build_weighted_graph(coords_all, dist_scale, threshold=0.001):
    """
    Build a weight matrix using a Gaussian kernel.
    """
    # Compute the full pairwise distance matrix using the Haversine metric.
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
        coords_all: numpy array of shape (n_samples, 2), where each row is [latitude, longitude].
        sigma: Bandwidth parameter for the Gaussian kernel (in meters).
        threshold: Minimum weight for an edge to be included in the graph.
    """
    weight_matrix = build_weighted_graph(coords_all, sigma, threshold)

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
    epochs = range(1, len(train_mse_list) + 1)

    plt.figure(figsize=(14, 6))

    # Plot MSE metrics
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

    # Plot R2 metrics
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
    
def calculate_correlation_coefficient_all_timepoints(data, show_plot = False):
    corr_dict = {}
    
    for tp in range(1, 7): 
        stage = f"R{tp}"
        df = data[data['timepoint']==stage].dropna(subset=['Yield'])
        features = df.columns[4:]  # Select features from index 4 to the last column
        target = 'Yield' 
        
        corr_dict[stage] = {}
        for feature in features:
            # Compute Pearson correlation
            corr, _ = pearsonr(df[feature], df[target])
            corr_dict[stage][feature] = corr
    
        num_features = len(features)
        num_stat = 5   #mean, median, q1, q3, sum
        rows = int(num_features / num_stat)
        cols = num_stat
        
        if show_plot:
            plt.rcParams['font.family'] = 'Times New Roman'
            # Create a figure and a grid of subplots
            fig, axes = plt.subplots(rows, cols, figsize=(10, 8))  # Adjust size as needed
            axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
            
            # Loop through features and plot them
            for i, feature in enumerate(features):
                ax = axes[i]
                sns.regplot(x=df[feature], y=df[target], 
                            scatter_kws={"alpha": 0.5, "s": 18},
                            line_kws={"color": "red"}, ax=ax)
                
                # Set labels and title
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(f'{feature} r = {corr_dict[stage][feature]:.2f}', fontsize=10)
        
            # # Hide empty subplots if there are fewer than 25
            # for j in range(i + 1, len(axes)):
            #     fig.delaxes(axes[j])  # Remove unused subplots
            fig.suptitle(f'Pearson Correlation Coefficients for {stage}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    return corr_dict

def select_features_with_high_correlation(df, target='yield', threshold=0.5):
    
    if isinstance(df, dict):
        df = pd.DataFrame(df)
        
    exclude_cols = ['plot_id', 'irrigation_labels', 'irrigation']
    
    # Select numeric columns and exclude specified columns
    numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)
    df_numeric = df[numeric_cols]
    
    # Compute the Pearson correlation matrix
    corr_matrix = df_numeric.corr(method='pearson')
    
    # Get correlation of each feature with the target, dropping the target itself
    target_corr = corr_matrix[target].drop(target)
    
    # Select features with absolute correlation above the threshold
    selected_features = target_corr[abs(target_corr) > threshold].index.tolist()
    
    return selected_features



def save_best_model_predictions(data_dict, train_idx, val_idx, test_idx, best_preds_all, filepath):
    """
    Save best model predictions with irrigation treatment for train, validation, and test splits.

    Parameters:
        data_dict (dict): Dictionary containing data, including 'irrigation_labels' and 'yield'.
        train_idx (array-like): Indices corresponding to the training split.
        val_idx (array-like): Indices corresponding to the validation split.
        test_idx (array-like): Indices corresponding to the test split.
        best_preds_all (np.array): Predictions for all samples from the best model.
        filename (str): Name of the CSV file to save the results.

    Returns:
        pd.DataFrame: Combined DataFrame of all splits.
    """
    # Create DataFrame for training split
    train_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][train_idx],
        'irrigation_trt': data_dict['irrigation_labels'][train_idx],
        'yield_true': data_dict['yield'][train_idx],
        'yield_pred': best_preds_all[train_idx].squeeze()
    })
    train_df['split'] = 'train'

    # Create DataFrame for validation split
    val_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][val_idx],
        'irrigation_trt': data_dict['irrigation_labels'][val_idx],
        'yield_true': data_dict['yield'][val_idx],
        'yield_pred': best_preds_all[val_idx].squeeze()
    })
    val_df['split'] = 'validation'

    # Create DataFrame for test split
    test_df = pd.DataFrame({
        'plot_id': data_dict['plot_id'][test_idx],
        'irrigation_trt': data_dict['irrigation_labels'][test_idx],
        'yield_true': data_dict['yield'][test_idx],
        'yield_pred': best_preds_all[test_idx].squeeze()
    })
    test_df['split'] = 'test'

    # Combine all splits into one DataFrame and save to CSV
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df.to_csv(filepath, index=False)
    #print(f"Saved best model prediction results with irrigation treatment for train, validation, and test sets to '{filename}'.")

    return all_df

def calculate_metrics(y_test, test_preds):
    mse = mean_squared_error(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    corr = np.corrcoef(np.ravel(y_test), np.ravel(test_preds))[0, 1]
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}

def custom_loss(y_pred, y_true, loss_method='logcosh', huber_beta=1.0):
    """
    Compute the loss between predictions and true values based on the specified loss method.
    
    Parameters:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.
        loss_method (str): The loss function to use. Supported options are:
            - 'logcosh': Log-cosh loss.
            - 'mse': Mean squared error.
            - 'mae': Mean absolute error.
            - 'huber': Huber loss.
        huber_beta (float): The beta parameter for the Huber loss (only used if loss_method is 'huber').
    
    Returns:
        torch.Tensor: The computed loss.
    
    Raises:
        ValueError: If an unsupported loss_method is provided.
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


