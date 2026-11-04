"""
Main script for training and evaluating the breast tumor response prediction model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from config.config import *
from data.dataset import BreastDataset, CustomAugmentation
from models.resnet import resnet18
from models.siamese import SiameseNetwork
from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.kfold import train_model_with_kfold


def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    # Set device
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() and not torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create main dataset
    print("Loading main dataset...")
    full_dataset = BreastDataset(
        PRE_TREATMENT_DIR, POST_TREATMENT_DIR, METADATA_CSV,
        PRE_SEG_DIR, POST_SEG_DIR,
        target_shape=IMAGE_SHAPE,
        transform=CustomAugmentation(flip_prob=0.5, rotate_prob=0.5, max_rotate=10)
    )

    # Calculate class weights
    labels = [full_dataset[i][5].item() for i in range(len(full_dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples")

    # Ensure we have exactly two classes
    if len(unique) != 2:
        raise ValueError(f"Expected 2 classes, but found {len(unique)} classes")

    class_weights = compute_class_weight('balanced', classes=unique, y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights:", class_weights)

    # Oversample minority class
    oversampled_pairs = full_dataset.oversample_minority_class()
    print(f"Number of samples after oversampling: {len(oversampled_pairs)}")
    full_dataset.file_pairs = oversampled_pairs

    # Create loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Create external validation datasets
    print("Loading external validation datasets...")
    external_dataset = BreastDataset(
        EXTERNAL_PRE_DIR, EXTERNAL_POST_DIR, EXTERNAL_CSV,
        EXTERNAL_PRE_SEG_DIR, EXTERNAL_POST_SEG_DIR,
        target_shape=IMAGE_SHAPE
    )

    extra_external_dataset = BreastDataset(
        EXTRA_EXTERNAL_PRE_DIR, EXTRA_EXTERNAL_POST_DIR, EXTRA_EXTERNAL_CSV,
        EXTRA_EXTERNAL_PRE_SEG_DIR, EXTRA_EXTERNAL_POST_SEG_DIR,
        target_shape=IMAGE_SHAPE
    )

    # Initialize ResNet18 model
    base_model = resnet18(
        sample_input_W=IMAGE_SHAPE[0],
        sample_input_H=IMAGE_SHAPE[1],
        sample_input_D=IMAGE_SHAPE[2],
        shortcut_type='B',
        no_cuda=False,
        num_seg_classes=2
    )

    # Choose training method: k-fold cross-validation or single train/validation split
    use_kfold = True
    
    if use_kfold:
        # Use k-fold cross-validation
        print(f"Starting {NUM_FOLDS}-fold cross-validation...")
        train_model_with_kfold(
            full_dataset, external_dataset, extra_external_dataset,
            base_model, PRETRAINED_PATH, criterion,
            NUM_EPOCHS, device, SAVE_DIR, UNFREEZE_INTERVAL, EARLY_UNFREEZE_EPOCHS,
            model_name="ResNet18", learning_rate=LEARNING_RATE, save_interval=SAVE_INTERVAL,
            batch_size=BATCH_SIZE, n_splits=NUM_FOLDS
        )
    else:
        # Use single train/validation split
        from sklearn.model_selection import train_test_split
        
        # Split dataset into train and validation sets
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=[full_dataset[i][5].item() for i in indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices),
            num_workers=4
        )
        
        val_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=4
        )
        
        external_loader = DataLoader(
            external_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        extra_external_loader = DataLoader(
            extra_external_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        # Initialize model
        model = SiameseNetwork(base_model, PRETRAINED_PATH).to(device)
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Train model
        print("Starting training with single train/validation split...")
        performance, results = train_model(
            train_loader, val_loader, external_loader, extra_external_loader,
            model, criterion, optimizer, NUM_EPOCHS, device, SAVE_DIR,
            UNFREEZE_INTERVAL, EARLY_UNFREEZE_EPOCHS, model_name="ResNet18",
            learning_rate=LEARNING_RATE, save_interval=SAVE_INTERVAL
        )
        
        print(f"Training complete. Best performance: {performance:.4f}")


if __name__ == '__main__':
    main() 