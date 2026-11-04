"""
K-fold cross-validation utilities for breast tumor response prediction model.
"""

import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.train import train_model


def train_model_with_kfold(dataset, external_dataset, extra_external_dataset, 
                          resnet_model, pretrained_path, criterion, 
                          num_epochs, device, save_dir,
                          unfreeze_interval=10, early_unfreeze_epochs=5, 
                          model_name="ResNet18", learning_rate=1e-4,
                          save_interval=10, batch_size=16, n_splits=5,
                          kfold_splitter=None):
    """
    Train the model using k-fold cross-validation.
    
    Args:
        dataset (Dataset): Main dataset for training and validation
        external_dataset (Dataset): External dataset for validation
        extra_external_dataset (Dataset): Additional external dataset for validation
        resnet_model: ResNet model for feature extraction
        pretrained_path (str): Path to pretrained weights
        criterion (Loss): Loss function
        num_epochs (int): Number of epochs to train
        device (torch.device): Device to use for training
        save_dir (str): Directory to save models and results
        unfreeze_interval (int): Number of epochs between unfreezing layers
        early_unfreeze_epochs (int): Number of epochs without improvement before early unfreezing
        model_name (str): Name of the model for saving
        learning_rate (float): Learning rate for the optimizer
        save_interval (int): Number of epochs between saving models
        batch_size (int): Batch size for training and validation
        n_splits (int): Number of folds for cross-validation
        kfold_splitter: KFold splitter for cross-validation
        
    Returns:
        pd.DataFrame: DataFrame with all results from k-fold cross-validation
    """
    from models.siamese import SiameseNetwork
    
    if kfold_splitter is None:
        from sklearn.model_selection import KFold
        kfold_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold_splitter.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4)
        external_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        extra_external_loader = DataLoader(extra_external_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Reinitialize model for each fold
        model = SiameseNetwork(resnet_model, pretrained_path).to(device)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Train the model
        fold_performance, fold_results = train_model(
            train_loader, val_loader, external_loader, extra_external_loader, 
            model, criterion, optimizer,
            num_epochs, device, save_dir, unfreeze_interval, early_unfreeze_epochs,
            model_name=f"{model_name}_fold_{fold}", learning_rate=learning_rate, 
            save_interval=save_interval, fold=fold
        )

        # Add fold information to results
        for result in fold_results:
            result['Fold'] = fold

        all_results.extend(fold_results)

        print(f"Fold {fold} Best Performance: {fold_performance:.4f}")

    # Save all fold results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(save_dir, f'training_results_{model_name}_{n_splits}fold.csv'), index=False)
    print(f"\nAll {n_splits}-fold cross-validation results saved to '{os.path.join(save_dir, f'training_results_{model_name}_{n_splits}fold.csv')}'")
    
    return all_results_df 