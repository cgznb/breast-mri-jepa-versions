"""
Configuration settings for breast tumor response prediction model.
"""

import os
import torch

# Data paths
PRE_TREATMENT_DIR = '/path/to/pre_treatment/scans'
POST_TREATMENT_DIR = '/path/to/post_treatment/scans'
PRE_SEG_DIR = '/path/to/pre_treatment/segmentations'
POST_SEG_DIR = '/path/to/post_treatment/segmentations'
METADATA_CSV = '/path/to/metadata.csv'

# External validation datasets
EXTERNAL_PRE_DIR = '/path/to/external/pre_treatment'
EXTERNAL_POST_DIR = '/path/to/external/post_treatment'
EXTERNAL_PRE_SEG_DIR = '/path/to/external/pre_segmentations'
EXTERNAL_POST_SEG_DIR = '/path/to/external/post_segmentations'
EXTERNAL_CSV = '/path/to/external_metadata.csv'

# Additional external validation dataset
EXTRA_EXTERNAL_PRE_DIR = '/path/to/extra_external/pre_treatment'
EXTRA_EXTERNAL_POST_DIR = '/path/to/extra_external/post_treatment'
EXTRA_EXTERNAL_PRE_SEG_DIR = '/path/to/extra_external/pre_segmentations'
EXTRA_EXTERNAL_POST_SEG_DIR = '/path/to/extra_external/post_segmentations'
EXTRA_EXTERNAL_CSV = '/path/to/extra_external_metadata.csv'

# Model parameters
IMAGE_SHAPE = (150, 150, 150)  # Target shape for the 3D images
PRETRAINED_PATH = '/path/to/pretrained/resnet18.pth'  # Path to pretrained weights

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 700
LEARNING_RATE = 1e-5
UNFREEZE_INTERVAL = 100  # Number of epochs between unfreezing layers
EARLY_UNFREEZE_EPOCHS = 5  # Early unfreeze if no improvement for this many epochs
SAVE_INTERVAL = 5  # Save model every N epochs
NUM_FOLDS = 5  # Number of folds for cross-validation

# Output directory for saving models and results
SAVE_DIR = './saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_ID = 0  # Set to specific GPU ID if multiple GPUs are available

# Model architecture parameters
FEATURE_DIM = 512
NUM_HEADS = 8
TRANSFORMER_LAYERS = 2
MAX_REGIONS = 3  # Maximum number of sub-regions to analyze 