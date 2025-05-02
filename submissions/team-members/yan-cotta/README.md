# Week 2 Submission: Data Preprocessing by Yan

## Tasks Completed
- **Dataset Retrieval**: Extracted the Kaggle dataset using `extract.py`.
- **Image Verification**: Checked for corrupt images with `check_images.py` (no issues found).
- **Class Imbalance Analysis**: Analyzed class distribution and visualized it in `class_distribution.png` using `analyze_classes.py`.
- **Data Preprocessing**: Split the dataset into training (10,414 samples) and validation (2,610 samples) sets with `data_preprocessing.py`.

## Files
- `extract.py`: Extracts the dataset zip file.
- `check_images.py`: Verifies image integrity.
- `analyze_classes.py`: Analyzes and plots class distribution.
- `class_distribution.png`: Visualization of class distribution.
- `data_preprocessing.py`: Splits dataset into train/validation sets.

## Notes
- **Training Samples**: 10,414
- **Validation Samples**: 2,610
- **Class Folders**: Confirmed in `split_dataset/train` and `split_dataset/val` with no "Invalid" folder issues
- Check `class_distribution.png` for specific class imbalances (e.g., some classes may have zero samples)

## How to Reproduce This Work

To replicate the preprocessing steps performed with the provided scripts, follow these instructions. Since the dataset is too large to include in the repository, you'll need to download it separately and process it using the scripts in this project.

### Download the Dataset

1. Obtain the Kaggle dataset (`new-bangladeshi-crop-disease.zip`) from [Kaggle Dataset](https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease)
2. Place the zip file in the project root directory (e.g., `~/Documents/Git/SDS-CP028-smart-leaf`)

### Process the Dataset

1. **Extract the Dataset**:
    ```bash
    python extract.py
    ```
    This extracts the contents of `new-bangladeshi-crop-disease.zip` into the `dataset/` folder

2. **Verify Images**:
    ```bash
    python check_images.py
    ```
    This scans the `dataset/` folder for corrupt images and removes them

3. **Analyze Class Distribution**:
    ```bash
    python analyze_classes.py
    ```
    This generates `class_distribution.png`, visualizing the number of images per class

4. **Split the Dataset**:
    - Remove non-class folders if present (e.g., `rm -r dataset/Invalid` or `rm dataset/Info.txt`)
    - Run:
      ```bash
      python data_preprocessing.py
      ```
    This splits the dataset into `split_dataset/train` and `split_dataset/val` with an 80/20 ratio

### Prerequisites
```bash
pip install -r requirements.txt
```
