"""
Evaluation utilities for breast tumor response prediction model.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F


def evaluate_model(data_loader, model, criterion, device, desc="Evaluating"):
    """
    Evaluate the model on a dataset.
    
    Args:
        data_loader (DataLoader): DataLoader for evaluation data
        model (nn.Module): Model to evaluate
        criterion (Loss): Loss function
        device (torch.device): Device to use for evaluation
        desc (str): Description for progress bar
        
    Returns:
        tuple: (avg_loss, auc, accuracy) for the evaluation
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for pre_img, post_img, pre_regions, post_regions, diff_time, batch_labels in tqdm(data_loader, desc=desc):
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            pre_regions = pre_regions.to(device)
            post_regions = post_regions.to(device)
            diff_time = diff_time.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(pre_img, post_img, pre_regions, post_regions, diff_time)
            loss = criterion(outputs, batch_labels.long())

            total_loss += loss.item()
            predictions.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Convert predictions to binary labels
    binary_predictions = (np.array(predictions) > 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(labels, predictions)  # Use continuous predictions for AUC
    acc = accuracy_score(labels, binary_predictions)  # Use binary predictions for accuracy

    return avg_loss, auc, acc


def get_prediction_range(model, loader, device):
    """
    Get the range of prediction values from the model.
    
    This is useful for calibrating the model's prediction threshold.
    
    Args:
        model (nn.Module): Model to evaluate
        loader (DataLoader): DataLoader for evaluation data
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: (min_prediction, max_prediction) for the dataset
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for pre_img, post_img, pre_regions, post_regions, diff_time, _ in loader:
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            pre_regions = pre_regions.to(device)
            post_regions = post_regions.to(device)
            diff_time = diff_time.to(device)

            outputs = model(pre_img, post_img, pre_regions, post_regions, diff_time)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)
            
    return np.min(all_predictions), np.max(all_predictions) 