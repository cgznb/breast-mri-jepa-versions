# MRI Spatiotemporal Interaction Model for Early Noninvasive Predicting Neoadjuvant Chemotherapy Response in Breast Cancer: multicentre study

A deep learning model for predicting pathologic complete response (pCR) in breast cancer patients using pre- and post-treatment MRI scans.

## Features

- Siamese network architecture based on ResNet18
- Integration of temporal information between pre- and post-treatment scans
- Sub-region analysis for more detailed feature extraction
- Multi-head temporal attention mechanism

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/breast_tumor_response_prediction.git
cd breast_tumor_response_prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the correct format
2. Configure the parameters in `config/config.py`
3. Run the training script:
```bash
python main.py
```

## Data Structure

The model expects pre- and post-treatment MRI scans along with segmentation masks for sub-regions. Data should be organized as follows:

```
data/
├── pre_treatment/
│   └── patient_id_scan.nii.gz
├── post_treatment/
│   └── patient_id_scan.nii.gz
├── pre_segmentation/
│   └── patient_id_scan_RadiomicsClass.nii.gz
├── post_segmentation/
│   └── patient_id_scan_RadiomicsClass.nii.gz
└── metadata.csv
```

The CSV file should contain columns for patient ID, PCR status (0 or 1), and time difference between scans.

## Model Architecture

The model uses a Siamese network with ResNet18 backbone for feature extraction from pre- and post-treatment images. Temporal transformer blocks enhance the representation of changes between timepoints. Sub-region analysis provides more detailed features of tumor regions.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 