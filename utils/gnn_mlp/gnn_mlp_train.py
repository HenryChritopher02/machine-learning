import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from utils.model_seed import set_seed
# import umap
# from sklearn.manifold import TSNE

set_seed(seed=42)

def forward(model, mlp1, combined_mlp_model, batch, batch_numerical_features, device='cuda'):
    batch.x, batch.edge_index, batch.batch = batch.x.to(device).float(), batch.edge_index.to(device), batch.batch.to(device)
    batch_numerical_features = batch_numerical_features.to(device).float()
    
    gin_output = model(batch.x, batch.edge_index, batch.batch)
    mlp_output = mlp1(batch_numerical_features)
    combined_features = torch.cat((gin_output, mlp_output), dim=1)

    final_output = combined_mlp_model(combined_features)
    return final_output, combined_features, gin_output, mlp_output #combined_features

def save_checkpoint(model, mlp1, combined_mlp_model, optimizer, epoch, loss, model_path, mlp_path, hybrid_path):
    # Save the model states and optimizer for all models
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': mlp1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, mlp_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': combined_mlp_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, hybrid_path)

def train_hybrid_model(model, mlp1, combined_mlp_model, graph_train_loader, numerical_train_loader, graph_valid_loader, numerical_valid_loader, epochs=100, learning_rate=0.01, patience=None, device='cpu', model_path='best_gnn.pth', mlp_path='best_mlp.pth', hybrid_path='best_hybrid.pth'):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(mlp1.parameters()) + list(combined_mlp_model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, min_lr=1e-6)

    train_losses, valid_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    model.to(device)
    mlp1.to(device)
    combined_mlp_model.to(device)

    for epoch in range(epochs):
        model.train()
        mlp1.train()
        combined_mlp_model.train()

        running_loss = 0.0
        for (graph_batch, numerical_batch) in zip(graph_train_loader, numerical_train_loader):
            num_features, labels = numerical_batch
            graph_batch, num_features, labels = graph_batch.to(device), num_features.to(device), labels.to(device)

            output, _, _, _ = forward(model, mlp1, combined_mlp_model, graph_batch, num_features, device)
            loss = criterion(output, graph_batch.y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(graph_train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        mlp1.eval()
        combined_mlp_model.eval()

        valid_loss = 0.0
        with torch.no_grad():
            for (graph_batch, numerical_batch) in zip(graph_valid_loader, numerical_valid_loader):
                num_features, labels = numerical_batch
                graph_batch, num_features, labels = graph_batch.to(device), num_features.to(device), labels.to(device)

                val_output, _, _, _ = forward(model, mlp1, combined_mlp_model, graph_batch, num_features, device)
                val_loss = criterion(val_output, graph_batch.y.unsqueeze(1))
                valid_loss += val_loss.item()

        # Calculate average validation loss
        valid_loss /= len(graph_valid_loader)
        valid_losses.append(valid_loss)

        scheduler.step(valid_loss)

        # Print the loss for this epoch
        if epoch % 10 == 0:
          print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch
            save_checkpoint(model, mlp1, combined_mlp_model, optimizer, epoch, valid_loss, model_path, mlp_path, hybrid_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # After training, load the best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    mlp1.load_state_dict(torch.load(mlp_path)['model_state_dict'])
    combined_mlp_model.load_state_dict(torch.load(hybrid_path)['model_state_dict'])

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.show()

    return model, mlp1, combined_mlp_model

def evaluate_hybrid_model(model, mlp1, combined_mlp_model, graph_data_loader, numerical_data_loader, device='cpu'):
    model.to(device).eval()
    mlp1.to(device).eval()
    combined_mlp_model.to(device).eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad(): # no_grad inference_mode
        for (graph_batch, numerical_batch) in zip(graph_data_loader, numerical_data_loader):
            # Extract numerical features and labels from numerical_batch
            num_features, labels = numerical_batch
            
            # Move tensors to the device
            graph_batch = graph_batch.to(device)
            num_features = num_features.to(device)
            labels = labels.to(device)
            predictions, _, _, _ = forward(model, mlp1, combined_mlp_model, graph_batch, num_features, device)
            all_predictions.append(predictions.cpu().squeeze().numpy())
            all_targets.append(graph_batch.y.cpu().numpy())

    # Convert predictions and targets to NumPy arrays for sklearn metrics
    predictions_np = np.concatenate(all_predictions)
    y_np = np.concatenate(all_targets)

    # Compute metrics
    mse = mean_squared_error(y_np, predictions_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_np, predictions_np)
    r2 = r2_score(y_np, predictions_np)
    pearson_corr, _ = pearsonr(y_np, predictions_np)
    spearman_corr, _ = spearmanr(y_np, predictions_np)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared (R2): {r2:.4f}')
    print(f'Pearson correlation: {pearson_corr:.4f}')
    print(f'Spearman correlation: {spearman_corr:.4f}')
    
def predict_hybrid(model, mlp1, combined_mlp_model, graph_data_loader, numerical_data_loader, output_csv_path=None, device='cpu'):
    model.to(device)
    mlp1.to(device)
    combined_mlp_model.to(device)
    model.eval()
    mlp1.eval()
    combined_mlp_model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for (graph_batch, numerical_batch) in zip(graph_data_loader, numerical_data_loader):
            # Extract numerical features and labels from numerical_batch
            num_features, labels = numerical_batch
            
            # Move tensors to the device
            graph_batch = graph_batch.to(device)
            num_features = num_features.to(device)
            labels = labels.to(device)
            predictions, _, _, _ = forward(model, mlp1, combined_mlp_model, graph_batch, num_features, device)
            all_predictions.append(predictions.cpu().squeeze().numpy())
            all_targets.append(graph_batch.y.cpu().numpy())

    # Convert predictions and targets to NumPy arrays for sklearn metrics
    predictions_np = np.concatenate(all_predictions)
    y_np = np.concatenate(all_targets)

    # Compute metrics
    r2 = r2_score(y_np, predictions_np)
    mae = mean_absolute_error(y_np, predictions_np)
    mse = mean_squared_error(y_np, predictions_np)
    rmse = np.sqrt(mse)
    # spearmanr and pearsonr return (correlation, p-value)
    spearman_corr, _ = spearmanr(y_np, predictions_np)
    pearson_corr, _ = pearsonr(y_np, predictions_np)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared (R2): {r2:.4f}')
    print(f'Pearson correlation: {pearson_corr:.4f}')
    print(f'Spearman correlation: {spearman_corr:.4f}')
    
    # Create a DataFrame with one row where columns are the metrics
    metrics_dict = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'spearman': spearman_corr,
        'pearson': pearson_corr
    }
    results_df = pd.DataFrame(metrics_dict, index=[0])
    
    # Save the metrics to CSV only if output_csv_path is provided
    if output_csv_path is not None:
        results_df.to_csv(output_csv_path, index=False)
    
    return results_df

def predict_pic50_hybrid(model, mlp1, combined_mlp_model, graph_data_loader, numerical_data_loader, device='cpu'):
    model.to(device)
    mlp1.to(device)
    combined_mlp_model.to(device)

    model.eval()
    mlp1.eval()
    combined_mlp_model.eval()

    # Lists to store information per sample
    all_smiles = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for (graph_batch, numerical_batch) in zip(graph_data_loader, numerical_data_loader):
            # Extract numerical features and labels from numerical_batch
            num_features, labels = numerical_batch
            
            # Move tensors to the device
            graph_batch = graph_batch.to(device)
            num_features = num_features.to(device)
            labels = labels.to(device)
            predictions, _, _, _ = forward(model, mlp1, combined_mlp_model, graph_batch, num_features, device)
            predictions_np = predictions.cpu().squeeze().numpy()
            targets_np = graph_batch.y.cpu().numpy()
            
            # Append batch results: ensure the ordering matches that of predictions/targets
            all_smiles.extend(graph_batch.smiles)
            all_predictions.append(predictions_np)
            all_targets.append(targets_np)

    # Flatten predictions and targets across batches
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Create a DataFrame with one row per sample
    df = pd.DataFrame({
        'standardized_smiles': all_smiles,
        'actual_pIC50': all_targets,
        'predicted_pIC50': all_predictions
    })
    
    return df

# def get_representations(model: torch.nn.Module,
#                         mlp1: torch.nn.Module,
#                         combined_mlp_model: torch.nn.Module,
#                         graph_data_loader: torch.utils.data.DataLoader,
#                         numerical_data_loader: torch.utils.data.DataLoader,
#                         device: str = 'cpu') -> tuple:
#     """
#     Extracts and concatenates learned representations from batches and retrieves targets from graph_batch.y.

#     Args:
#         model (torch.nn.Module): Graph-based model.
#         mlp1 (torch.nn.Module): MLP model for processing numerical data.
#         combined_mlp_model (torch.nn.Module): Model that combines features.
#         graph_data_loader (torch.utils.data.DataLoader): DataLoader yielding graph data batches with a 'y' attribute for targets.
#         numerical_data_loader (torch.utils.data.DataLoader): DataLoader yielding (features, labels) for numerical data.
#         device (str, optional): Device identifier (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

#     Returns:
#         tuple: A tuple (representations, targets) where:
#             - representations (np.ndarray): Concatenated array of learned representations from all batches.
#             - targets (np.ndarray): Concatenated array of target values from graph_batch.y.
#     """
#     # Move models to the device and set to evaluation mode.
#     for mdl in (model, mlp1, combined_mlp_model):
#         mdl.to(device)
#         mdl.eval()

#     all_representations = []
#     mol_representations = []
#     ds_representations = []
#     all_targets = []

#     with torch.no_grad():
#         for graph_batch, numerical_batch in zip(graph_data_loader, numerical_data_loader):
#             # Unpack numerical features (labels are not used here)
#             num_features, _ = numerical_batch

#             # Transfer data to the correct device.
#             graph_batch = graph_batch.to(device)
#             num_features = num_features.to(device)

#             # Forward pass through the models.
#             # Assumes `forward` returns a tuple: (predictions, combined_features).
#             _, combined_features, gin_output, mlp_output = forward(model, mlp1, combined_mlp_model,
#                                            graph_batch, num_features, device)

#             # Append representations.
#             all_representations.append(combined_features.cpu().numpy())
#             mol_representations.append(gin_output.cpu().numpy())
#             ds_representations.append(mlp_output.cpu().numpy())

#             # Extract and append targets from graph_batch.y.
#             all_targets.append(graph_batch.y.cpu().numpy())

#     # Concatenate representations and targets from all batches.
#     representations_np = np.concatenate(all_representations, axis=0)
#     mol_np = np.concatenate(mol_representations, axis=0)
#     ds_np = np.concatenate(ds_representations, axis=0)
#     targets_np = np.concatenate(all_targets, axis=0)
#     return representations_np, mol_np, ds_np, targets_np

# def plot_tnse_representation(representations: np.ndarray,
#                              targets: np.ndarray = None,
#                              epoch_number: int = None,
#                              output_plot_path: str = 'tnse_plot.png') -> None:
#     """
#     Tạo và hiển thị trực quan hóa TNSE của các biểu diễn học được.

#     Args:
#         representations (np.ndarray): Mảng các biểu diễn học được với kích thước (n_samples, representation_dim).
#         targets (np.ndarray, optional): Giá trị target để tô màu các điểm. Mặc định là None.
#         epoch_number (int, optional): Số epoch để hiển thị trong tiêu đề biểu đồ. Mặc định là None.
#         output_plot_path (str, optional): Đường dẫn lưu biểu đồ. Mặc định là 'tnse_plot.png'.
#     """

#     # Thực hiện giảm chiều dữ liệu bằng TNSE.
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=15, random_state=42)
#     representations = pca.fit_transform(representations)
#     tsne = TSNE(n_components=2,
#             perplexity=50,
#             learning_rate=200,
#             init='pca',
#             early_exaggeration=30,
#             random_state=42)
#     reduced_representations = tsne.fit_transform(representations)
#     # tnse = TSNE(n_components=2, random_state=42, perplexity=200)
#     # reduced_representations = tnse.fit_transform(representations)

#     plt.figure(figsize=(8, 6))
#     if targets is not None:
#         scatter = plt.scatter(reduced_representations[:, 0],
#                               reduced_representations[:, 1],
#                               c=targets, cmap='jet', s=10)
#         plt.colorbar(scatter, label='Target Value')
#     else:
#         plt.scatter(reduced_representations[:, 0],
#                     reduced_representations[:, 1],
#                     s=10)

#     plt.xlabel('TNSE Dimension 1')
#     plt.ylabel('TNSE Dimension 2')
    
#     title = 'TNSE Visualization of Learned Representations'
#     if epoch_number is not None:
#         title += f' - Epoch {epoch_number}'
#     plt.title(title)

#     plt.savefig(output_plot_path)
#     plt.show()