"""
Training utilities for breast tumor response prediction model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.evaluate import evaluate_model


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        loader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        criterion (Loss): Loss function
        device (torch.device): Device to use for training
        
    Returns:
        tuple: (average_loss, accuracy, auc) for the epoch
    """
    model.train()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for pre_img, post_img, pre_regions, post_regions, diff_time, labels in tqdm(loader, desc="Training"):
        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        pre_regions = pre_regions.to(device)
        post_regions = post_regions.to(device)
        diff_time = diff_time.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pre_img, post_img, pre_regions, post_regions, diff_time)
        loss = criterion(outputs, labels.long())
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    binary_predictions = (all_predictions > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_predictions)
    auc = roc_auc_score(all_labels, all_predictions)

    return avg_loss, accuracy, auc


def train_model(train_loader, val_loader, external_loader, extra_external_loader, 
               model, criterion, optimizer, num_epochs, device, save_dir,
               unfreeze_interval=10, early_unfreeze_epochs=5, model_name="ResNet18", 
               learning_rate=1e-4, save_interval=10, fold=0):
    """
    Train the model with progressive unfreezing and validation on multiple datasets.
    
    Args:
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        external_loader (DataLoader): DataLoader for external validation data
        extra_external_loader (DataLoader): DataLoader for additional external validation data
        model (nn.Module): Model to train
        criterion (Loss): Loss function
        optimizer (Optimizer): Optimizer for training
        num_epochs (int): Number of epochs to train
        device (torch.device): Device to use for training
        save_dir (str): Directory to save models and results
        unfreeze_interval (int): Number of epochs between unfreezing layers
        early_unfreeze_epochs (int): Number of epochs without improvement before early unfreezing
        model_name (str): Name of the model for saving
        learning_rate (float): Learning rate for the optimizer
        save_interval (int): Number of epochs between saving models
        fold (int): Current fold number in k-fold cross-validation
        
    Returns:
        tuple: (best_performance, results) for the training run
    """
    best_overall_performance = float('-inf')

    # Calculate weights for train, validation, and two external validation sets
    total_samples = len(train_loader.dataset) + len(val_loader.dataset) + \
                    len(external_loader.dataset) + len(extra_external_loader.dataset)
    train_weight = len(train_loader.dataset) / total_samples
    val_weight = len(val_loader.dataset) / total_samples
    external_weight = len(external_loader.dataset) / total_samples
    extra_external_weight = len(extra_external_loader.dataset) / total_samples

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    # Define unfreezing order
    unfreeze_order = [
        ('fc', model.fc),
        ('attention', model.attention),
        ('layer4', model.resnet.layer4),
        ('layer3', model.resnet.layer3),
        ('layer2', model.resnet.layer2),
        ('layer1', model.resnet.layer1),
        ('conv1', model.resnet.conv1)
    ]

    # Initially freeze all layers
    for _, layer in unfreeze_order:
        for param in layer.parameters():
            param.requires_grad = False

    # Train only the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    current_unfrozen_layer = "fc"
    next_layer_to_unfreeze = 1  # Start with attention

    results = []
    epochs_without_improvement = 0

    # Create progress bar
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        print(f"Current unfrozen layer: {current_unfrozen_layer}")

        # Check if we need to unfreeze the next layer
        if epoch % unfreeze_interval == 0 and epoch > 0 and next_layer_to_unfreeze < len(unfreeze_order):
            layer_name, layer_to_unfreeze = unfreeze_order[next_layer_to_unfreeze]
            print(f"Unfreezing layer: {layer_name}")
            for param in layer_to_unfreeze.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                               weight_decay=1e-5)
            current_unfrozen_layer = layer_name
            next_layer_to_unfreeze += 1

        model.train()
        train_loss, train_accuracy, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on validation set and two external validation sets
        val_loss, val_auc, val_accuracy = evaluate_model(val_loader, model, criterion, device, desc="Validating")
        external_loss, external_auc, external_acc = evaluate_model(external_loader, model, criterion, device, desc="External Validation")
        extra_external_loss, extra_external_auc, extra_external_acc = evaluate_model(extra_external_loader, model, criterion, device, desc="Extra External Validation")

        # Calculate weighted overall performance (including two external validation sets)
        overall_auc = (0.2 * train_auc +
                       0.2 * val_auc +
                       0.3 * external_auc +
                       0.3 * extra_external_auc)
        overall_acc = (train_weight * train_accuracy +
                       val_weight * val_accuracy +
                       external_weight * external_acc +
                       extra_external_weight * extra_external_acc)
        overall_performance = overall_auc

        # Update progress bar description
        pbar.set_description(
            f"Best: {best_overall_performance:.4f} | Current: {overall_performance:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Ext AUC: {external_auc:.4f} | ExtraExt AUC: {extra_external_auc:.4f}")

        # Record results
        results.append({
            'Model': model_name,
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train AUC': train_auc,
            'Train Acc': train_accuracy,
            'Val Loss': val_loss,
            'Val AUC': val_auc,
            'Val Acc': val_accuracy,
            'External Loss': external_loss,
            'External AUC': external_auc,
            'External Acc': external_acc,
            'Extra External Loss': extra_external_loss,
            'Extra External AUC': extra_external_auc,
            'Extra External Acc': extra_external_acc,
            'Overall AUC': overall_auc,
            'Overall Acc': overall_acc,
            'Overall Performance': overall_performance,
            'Unfrozen Layer': current_unfrozen_layer
        })

        scheduler.step(overall_performance)  # Use overall_performance to adjust learning rate

        # Save model if both external validation metrics are above thresholds
        if external_auc >= 0.79 and extra_external_auc >= 0.6:
            save_path = os.path.join(save_dir, 
                f'model_{model_name}_fold_{fold}_epoch_{epoch+1}_ext_{external_auc:.3f}_extraext_{extra_external_auc:.3f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'performance': overall_performance,
                'external_auc': external_auc,
                'extra_external_auc': extra_external_auc,
            }, save_path)
            print(f"\nSaving high-performance model - External AUC: {external_auc:.3f}, Extra External AUC: {extra_external_auc:.3f}")
            print(f"Model saved to: {save_path}")

        # Save the best model
        if overall_performance > best_overall_performance:
            best_overall_performance = overall_performance
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'performance': best_overall_performance,
            }, os.path.join(save_dir, f'best_model_{model_name}.pth'))
            print(f"Saved best model, overall performance: {best_overall_performance:.4f}")
        else:
            epochs_without_improvement += 1

        # Save model at regular intervals
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'performance': overall_performance,
            }, os.path.join(save_dir, f'model_{model_name}_epoch_{epoch + 1}.pth'))
            print(f"Saved model at epoch {epoch + 1}, overall performance: {overall_performance:.4f}")

        # Early unfreezing check
        if epochs_without_improvement >= early_unfreeze_epochs:
            if next_layer_to_unfreeze < len(unfreeze_order):
                layer_name, layer_to_unfreeze = unfreeze_order[next_layer_to_unfreeze]
                print(f"Early unfreezing layer due to lack of improvement: {layer_name}")
                for param in layer_to_unfreeze.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                   weight_decay=1e-5)
                epochs_without_improvement = 0
                current_unfrozen_layer = layer_name
                next_layer_to_unfreeze += 1
            else:
                print(
                    f"Early stopping triggered. No improvement for {early_unfreeze_epochs} epochs and all layers are unfrozen.")
                break

    # Save training results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, f'training_results_{model_name}.csv'), index=False)
    print(f"\nTraining results saved to '{os.path.join(save_dir, f'training_results_{model_name}.csv')}'")

    return best_overall_performance, results 