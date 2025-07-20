# NIR-Based Skin Cancer Classification Using Machine Learning

This project replicates and builds upon the research presented in the paper [“Skin cancer diagnosis using NIR spectroscopy data of skin lesions in vivo using machine learning algorithms” (arXiv:2401.01200)](https://arxiv.org/pdf/2401.01200). Using near-infrared (NIR) spectroscopy data, we aim to distinguish between malignant and benign skin lesions by identifying the most predictive wavelength intervals and building optimized machine learning models.

## 📁 Dataset

The dataset is provided in `Databrief.xls`, which will be included in this repository. It contains absorbance measurements between 900–1600 nm for various skin lesion types, with associated class labels. Each row represents a sample, and each column from the 5th onward represents a different NIR wavelength.

Class Breakdown:
- **Malignant**: Melanoma (MEL), Basal Cell Carcinoma (BCC), Epidermal Carcinoma (CEC)
- **Benign**: Nevus (NEV), Actinic Keratosis (ACK), Seborrheic Keratosis (SEK)

---

## 🧪 Methodology

### 1. **Preprocessing**
- Spectral data was **standardized using SNV (Standard Normal Variate)** normalization.
- Binary classification labels were created (`1` = malignant, `0` = benign).

### 2. **Sliding Window Feature Extraction**
Using a dynamic window size `w`, the following statistical features were computed over each spectral window:
- Mean, Median, Standard Deviation
- Kurtosis, Skewness
- Max, Min, Peak-to-Peak
- Root Mean Square (RMS)

These features were used to form the input for downstream ML models.

### 3. **Hyperparameter Tuning with Optuna**
`Optuna` was used to optimize:
- Window size for feature extraction
- Hyperparameters for LightGBM and XGBoost classifiers

### 4. **Data Balancing with SMOTE**
Due to class imbalance (270 malignant vs. 444 benign samples), **SMOTE** was used to oversample the minority class and improve model performance, especially for recall.

### 5. **Modeling**
Multiple models were trained and evaluated:
- **LightGBM** (best Optuna-tuned parameters)
- **Random Forest**
- **XGBoost**
- **SVM**
- **1D GAN + LightGBM** (experimental)

Each model was evaluated on:
- Accuracy
- Precision (malignant)
- Recall (malignant)
- F1 Score
- 5-fold cross-validation on F1

---

## 🤖 Model Performance (Post-SMOTE)

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| LightGBM      | 0.818    | 0.759     | 0.759  | 0.759    |
| XGBoost       | ~0.82    | ~0.78     | **0.80**  | ~0.79    |
| Random Forest | ~0.81    | ~0.76     | ~0.76  | ~0.76    |
| SVM           | ~0.79    | ~0.73     | ~0.75  | ~0.74    |
| GAN+LGBM      | *Lower*  | *Lower*   | *Lower*| *Lower*  |

- **XGBoost had the highest recall**, which is critical for medical diagnostics.
- **GAN-based synthetic augmentation** was attempted, but PCA filtering reduced its effectiveness.

---

## 🔬 Top Predictive Spectral Windows

| Window (nm)    | Feature | Importance | Models Shared |
|----------------|---------|------------|----------------|
| 30–36          | min     | High       | RF, XGBoost    |
| 34–40          | std     | High       | RF, XGBoost    |
| 49–55          | kurt    | High       | LGBM, SVM      |
| 6–12           | skew    | High       | LGBM, SVM      |
| 9–15           | ptp     | High       | LGBM, RF       |

These wavelength regions are strongly linked to:
- **Water & lipid absorption** (945–982 nm)
- **Protein/collagen structure** (1093–1149 nm)
- **Fatty acid and cell membrane differences** (1211–1242 nm)

---

## 🧠 Key Takeaways

- **Sliding-window statistical features** are highly effective for spectral classification.
- **Optuna HPO** greatly improves model performance by tuning both window size and ML parameters.
- **SMOTE** helps handle imbalance but care must be taken with model generalization.

---

## 📚 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, LightGBM, XGBoost)
- Optuna (Hyperparameter tuning)
- SMOTE (Imbalanced-learn)
- PyTorch (GAN implementation)
- Matplotlib (Visualization)

---

## 📝 Citation

Original Paper:  
> Silva, V. M., et al. *Skin cancer diagnosis using NIR spectroscopy data of skin lesions in vivo using machine learning algorithms.* arXiv:2401.01200 [cs.LG], 2024.  
> [Link to paper](https://arxiv.org/pdf/2401.01200)

---




