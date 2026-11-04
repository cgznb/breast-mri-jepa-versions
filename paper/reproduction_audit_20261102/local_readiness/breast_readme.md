
# BreastDCEDL - 2,072 Breast Cancer Patients

Deep learning-ready dataset of pretreatment 3D dynamic contrast-enhanced MRI (DCE-MRI) scans from **2,072 breast cancer patients** across three clinical trials: **I-SPY2** (n=982), **I-SPY1** (n=172), and **Duke** (n=918). Features standardized 3D NIfTI volumes, tumor segmentations, and harmonized clinical metadata designed for reproducible research in breast cancer imaging analysis.
Nifti files preserve original DICOM intensity values (converted from uint16 to float64) without any preprocessing, normalization, or manipulation. All medical data processing code is available in the dataset-specific subdirectories for full transparency.

**Demo:** https://colab.research.google.com/github/naomifridman/BreastDCEDL/blob/main/BrestDCEDL_zenodo_demo.ipynb

# 📄 Citation
```bibtex
@article{fridman2026breastdcedl,
  author    = {Fridman, N. and Solway, B. and Fridman, T. and others},
  title     = {BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients},
  journal   = {Scientific Data},
  year      = {2026},
  doi       = {10.1038/s41597-026-06589-6},
  url       = {https://doi.org/10.1038/s41597-026-06589-6}
}
```
![fusion_grid_breastdcedl](https://github.com/naomifridman/BreastDCEDL/blob/main/images/fusion_grid_with_row_titles_fixed_3.png?raw=true)

### BreastDCEDL MinCrop Version

The MinCrop version provides three methodicaly selected DCE-MRI time points (pre-contrast, early post-contrast, late post-contrast) cropped to 256×256 pixels around the main tumor. This version has been successfully used to train deep learning models achieving state-of-the-art results for pCR (AUC 0.94) and HER2 (AUC 0.74) prediction in published research. 

The dataset is suitable for a wide range of computational approaches including radiomics feature extraction, traditional machine learning, and deep learning. The standardized CSV metadata format with numerically encoded clinical variables and 3D NIfTI imaging files ensures compatibility with all major programming languages and medical imaging software. Examples in this repository demonstrate implementation in Python using PyTorch and HuggingFace infrastructure, though the data format supports any computational framework.

**Key features:**
- **3 tumor-centered scans per patient**: Pre-contrast, early post-contrast, late post-contrast
- **Standardized size**: All scans cropped to 256×256 pixels around the main tumor
- **Fully available on Zenodo**: [Download MinCrop Dataset](https://zenodo.org/records/15627233)
- **Clinical relevance**: These three time points are specifically selected by radiologists for tumor identification, characterization, and segmentation in clinical practice

![BreastDCEDL ser images](https://github.com/naomifridman/BreastDCEDL/blob/main/images/ser_images.png?raw=true)

### BreastDCEDL Full Version - avalebale on TCIA

The Full version preserves complete spatiotemporal information for researchers investigating advanced temporal dynamics, developing novel preprocessing methods, or requiring the full field of view. Contains all acquired time points (3-12) at original resolution, enabling deep exploration of contrast enhancement kinetics and spatial relationships beyond the tumor region.

**Availability:**
- **I-SPY1 (Full)**: [Download from Zenodo](https://zenodo.org/records/15627233)
- **I-SPY2 (Full)**: [Download from breastdcedl-ispy2 TCIA](https://www.cancerimagingarchive.net/analysis-result/breastdcedl-ispy2/)
- **Duke**: Download from [TCIA](https://www.cancerimagingarchive.net/) and convert using provided code

**DCE MRI 3D Data Organization**
![BreastDCEDL data structure](https://github.com/naomifridman/BreastDCEDL/blob/main/images/breastdcedl_data_structure.png?raw=true)
### Dataset Characteristics

#### MinCrop Version (256×256×varied slices, 3 time points each)
| Dataset | Patients | Avg Z-slices | Size (GB) |
|---------|----------|--------------|-----------|
| I-SPY1 | 172 | 28 | ~2 |
| I-SPY2 | 982 | 39 | ~16 |
| Duke | 918 | 25 | ~10 |
| **Total** | **2,072** | **32** | **~28** |

#### Full Version (Original Resolution)
| Dataset | Patients | Common Resolutions | Avg Z-slices | Avg Time Points | Size (GB) |
|---------|----------|-------------------|--------------|-----------------|-----------|
| I-SPY1 | 172 | 256×256 (91%), 512×512 (9%) | 59 | 3.4 | ~6 |
| I-SPY2 | 982 | 256×256 (78%), 320×320 (12%), other (10%) | 106 | 7.2 | ~54 |
| Duke | 918 | 512×512 (68%), 448×448 (28%), 320×320 (4%) | 170 | 4.5 | ~142 |
| **Total** | **2,072** | **Multiple (256-512)** | **131** | **5.7** | **~202** |
  

## 🎯 Benchmark 

The dataset was split into training, validation, and test sets using stratified random sampling to ensure balanced distributions of pCR, HER2, and HR status across all splits. This multi-label stratification maintains consistent class proportions, enabling fair model comparison and preventing bias from imbalanced sampling. Reported performance metrics are calculated on the held-out test set.

### Data Splits

| Split | pCR N | pCR+ | pCR− | HR N | HR+ | HR− | HER2 N | HER2+ | HER2− |
|-------|-------|------|------|------|-----|-----|--------|-------|-------|
| **Training** | 1099 | 322 | 777 | 1529 | 987 | 542 | 1528 | 345 | 1183 |
| **Validation** | 176 | 53 | 123 | 268 | 167 | 101 | 268 | 58 | 210 |
| **Test** | 177 | 53 | 124 | 271 | 173 | 98 | 269 | 56 | 213 |
| **Total** | 1452 | 428 | 1024 | 2068 | 1327 | 741 | 2065 | 459 | 1606 |

*N = number of patients with available labels for each biomarker

### Classification Tasks and Results

Three standardized classification tasks are provided for benchmarking:

| Task | Description | Distribution | Best Performance |
|------|-------------|--------------|-----------------|
| **pCR Prediction** | Pathological complete response to neoadjuvant therapy | 29.5% positive (n=428/1452) | AUC 0.94 (ViT, HR+/HER2−)¹ |
| **HER2 Status** | HER2 expression | 22.1% positive (n=458/2070) | AUC 0.74 (Dual-Attention ResNet)² |
| **HR Status** | Hormone receptor positivity | 64.2% positive (n=1327/2070) | - |

¹Results from [Fridman et al., 2025 - BreastDCEDL](https://doi.org/10.48550/arXiv.2506.12190)  
²Results from [Fridman & Goldstein, 2025 - Dual-Attention ResNet](https://arxiv.org/abs/2510.13897)

## 🏥 Dataset Details

### BreastDCEDL_ISPY2 (n=982)
- **Sequences**: 3-12 time points (typically 7)
- **Annotations**: Full 3D tumor segmentations
![Example from I-SPY1](https://github.com/naomifridman/BreastDCEDL/blob/main/images/spy2_image_grid_git1.png?raw=true)
### BreastDCEDL_ISPY1 (n=172)
- **Sequences**: 3-5 usable DCE scans
- **Annotations**: Full 3D tumor segmentations
![Example from I-SPY1](https://github.com/naomifridman/BreastDCEDL/blob/main/images/spy1_image_grid_git1.png?raw=true)
### Duke (n=916)
- **NAC subset**: 298 patients with pCR labels (only 31% received neoadjuvant chemotherapy)
- **Sequences**: 1 pre-contrast + 2-4 post-contrast scans
- **Annotations**: Bounding box of largest tumor (no full segmentation)
![Example from Duke](https://github.com/naomifridman/BreastDCEDL/blob/main/images/duke_image_grid_git1.png?raw=true)
## 🚀 Quick Start

### Option 1: Explore Sample Data (No Download Required)
Open `BreastDCEDL_demo.ipynb` to visualize and explore example data included in this repository.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naomifridman/BreastDCEDL/blob/main/BreastDCEDL_demo.ipynb)

### Option 2: Work with Full MinCrop Dataset
1. Download the MinCrop dataset from [Zenodo](https://zenodo.org/records/15627233)
2. Open `BreastDCEDL_demo_on_local_data_min_crop.ipynb`
3. Follow the notebook to explore and analyze the data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naomifridman/BreastDCEDL/blob/main/BreastDCEDL_demo_on_local_data_min_crop.ipynb)

## 🔬 Clinical Background

Dynamic Contrast-Enhanced MRI (DCE-MRI) is a key imaging technique for breast cancer evaluation. It captures tissue perfusion dynamics through sequential 3D scans before and after contrast agent administration. In breast cancer:
- **Enhancement patterns**: Malignant tumors typically show rapid initial enhancement followed by washout or plateau
- **Clinical protocol**: Radiologists analyze pre-contrast and multiple post-contrast phases (typically 3-12 time points)
- **Predictive value**: Enhancement dynamics correlate with treatment response and can predict pathological complete response (pCR) to neoadjuvant therapy

For detailed methodology, see [Fridman et al., 2025](https://doi.org/10.48550/arXiv.2506.12190)

## 📚 Citations

If you use the BreastDCEDL dataset or code in your research, please cite the article:

### Article Citation (Required)
```bibtex
@article{fridman2025breastdcedl,
  title={BreastDCEDL: A Comprehensive Breast Cancer DCE-MRI Dataset and Transformer Implementation for Treatment Response Prediction},
  author={Fridman, Naomi and Solway, Bubby and Fridman, Tomer and Barnea, Itamar and Goldstein, Anat},
  journal={arXiv preprint arXiv:2506.12190},
  year={2025},
  doi={10.48550/arXiv.2506.12190}
}
```

### Dataset Citation
```bibtex
@dataset{fridman2025breastdcedl_dataset,
  author       = {Fridman, Naomi and others},
  title        = {BreastDCEDL: Curated DCE-MRI Dataset},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15627233}
}
```
## 📁 Repository Contents

| File | Description |
|------|-------------|
| `BreastDCEDL_metadata.csv` | Metadata for the full BreastDCEDL dataset |
| `BreastDCEDL_metadata_min_crop.csv` | Metadata for the MinCrop version |
| `BreastDCEDL_demo.ipynb` | Visualize data examples included in this repository |
| `BreastDCEDL_demo_on_local_data_min_crop.ipynb` | Explore and work with MinCrop data after downloading from Zenodo |
| `BreastDCEDL_vit_predict.ipynb` | Predict pCR using trained ViT model |

**Directories:** `I-SPY1/`, `I-SPY2/`, and `DUKE/` contain dataset-specific code for preprocessing metadata, DICOM exploration, and conversion to NIfTI format.

## 🔗 Resources
- **Zenodo Repository**: [https://zenodo.org/records/15627233](https://zenodo.org/records/15627233)
  
### Original Data Sources
- **I-SPY1**: Newitt, D., Hylton, N., on behalf of the I-SPY 1 Network and ACRIN 6657 Trial Team (2016). *Breast DCE-MRI Data and Segmentations from Patients in the I-SPY 1/ACRIN 6657 Trials* [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK](https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK)

- **I-SPY2**: Li, W., Newitt, D. C., Gibbs, J., Wilmes, L. J., Jones, E. F., Arasu, V. A., ... & Hylton, N. M. (2022). *I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)* (Version 1) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85)

- **Duke**: Saha, A., Harowicz, M. R., Grimm, L. J., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations (Duke-Breast-Cancer-MRI)* [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.e3sv-re93](https://doi.org/10.7937/TCIA.e3sv-re93)


## 📝 License

Please refer to the original data sources for licensing information.
