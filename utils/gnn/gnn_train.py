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

set_seed(seed=42)

def train_model(model, train_loader, valid_loader, epochs=100, learning_rate=0.01, patience=None, device='cpu', save_path='best_gnn_model.pth'):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, min_lr=1e-6)

    # Lists to store the loss values
    train_losses = []
    valid_losses = []

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.to(device)  # Move model to the specified device

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch.to(device)
            output = model(batch.x.float(), batch.edge_index,  batch.batch)
            loss = criterion(output, batch.y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch.to(device)
                val_output = model(batch.x.float(), batch.edge_index, batch.batch)
                val_loss = criterion(val_output, batch.y.unsqueeze(1))
                valid_loss += val_loss.item()

        # Calculate average validation loss
        valid_loss /= len(valid_loader)

        scheduler.step(valid_loss)
        valid_losses.append(valid_loss)

        # Print the loss for this epoch
        if epoch % 10 == 0:
          print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_state = model.state_dict()  # Save the model state
            best_epoch = epoch
            patience_counter = 0
        elif patience is not None:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # After training, load the best model
    if best_model_state is not None:
        torch.save({'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, save_path)
    # After training, load the best model
    model.load_state_dict(best_model_state)
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.show()

    return model

def evaluate_model(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad(): # no_grad inference_mode
        for batch in data_loader:
            batch.to(device)
            predictions = model(batch.x.float(), batch.edge_index, batch.batch)
            all_predictions.append(predictions.cpu().squeeze().numpy())
            all_targets.append(batch.y.cpu().numpy())

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

def load_model(model, path='best_model.pth', device='cpu'):
    # Load the model state from the specified file
    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.to(device)  # Move the model to the specified device
    return model

def predict_gnn(model, data_loader, output_csv_path, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            # Forward pass: obtain predictions for this batch
            predictions = model(batch.x.float(), batch.edge_index, batch.batch)
            # Collect predictions and true values
            all_predictions.append(predictions.cpu().squeeze().numpy())
            all_targets.append(batch.y.cpu().numpy())

    # Concatenate all predictions and targets from the different batches
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
    
    # Save the metrics to CSV
    results_df.to_csv(output_csv_path, index=False)
    
    return results_df

def predict_pic50_gnn(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    
    # Lists to store information per sample
    all_smiles = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            # Forward pass: obtain predictions for this batch
            predictions = model(batch.x.float(), batch.edge_index, batch.batch)
            predictions_np = predictions.cpu().squeeze().numpy()
            targets_np = batch.y.cpu().numpy()
            
            # Append batch results: ensure the ordering matches that of predictions/targets
            all_smiles.extend(batch.smiles)
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