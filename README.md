# Midterm Machine Learning 2025

## 📋 Identitas Mahasiswa
- **Nama**: Muhammad Rafi A Syifaa Nugraha
- **Kelas**: TK-46-04

---

## 🎯 Deskripsi Project
Repository ini berisi implementasi end-to-end Machine Learning untuk UTS (Midterm) yang mencakup tiga pendekatan utama: **Classification**, **Regression**, dan **Clustering** menggunakan berbagai dataset.

---

## 📁 Struktur Repository
```
Midterm-Machine-Learning-2025/
│
├── README.md                              # Dokumentasi project
├── .gitignore                             # Git ignore file
│
├── midterm_transaction_data.ipynb         # Classification - Fraud Detection
├── midterm_regresi.ipynb                  # Regression - Prediction
├── midterm_clustering.ipynb               # Clustering - Segmentation
│
├── datasets/
│   ├── transaction/
│   │   ├── train_transaction.csv          # Dataset training fraud detection
│   │   └── test_transaction.csv           # Dataset testing fraud detection
│   ├── regresi/
│   │   └── midterm-regresi-dataset.csv    # Dataset untuk regression
│   └── clustering/
│       └── clusteringmidterm.csv          # Dataset untuk clustering
│
├── models/                                # Folder untuk menyimpan trained models
└── midterm_folder/                        # Folder tambahan
```

---

## 📊 Dataset yang Digunakan

### 1. Transaction Dataset (Classification)
- **File**: `train_transaction.csv`, `test_transaction.csv`
- **Tujuan**: Deteksi fraud pada transaksi
- **Target**: `isFraud` (Binary classification)

### 2. Regression Dataset
- **File**: `midterm-regresi-dataset.csv`
- **Tujuan**: Prediksi nilai numerik

### 3. Clustering Dataset
- **File**: `clusteringmidterm.csv`
- **Tujuan**: Segmentasi data/customer

---

## 🚀 Model yang Diimplementasikan

### 1. Classification (`midterm_transaction_data.ipynb`)
| Model | Metrics |
|-------|---------|
| Logistic Regression | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| Random Forest | Feature Importance, Accuracy, ROC-AUC |
| XGBoost | Precision-Recall, ROC-AUC |
| Neural Network (Deep Learning) | Loss curves, Accuracy, Confusion Matrix |

### 2. Regression (`midterm_regresi.ipynb`)
| Model | Metrics |
|-------|---------|
| Linear Regression | R², MSE, MAE |
| Ridge Regression | R², RMSE |
| Lasso Regression | R², MAE |
| Random Forest Regressor | R², Feature Importance |
| Gradient Boosting | R², RMSE, MAE |

### 3. Clustering (`midterm_clustering.ipynb`)
| Model | Metrics |
|-------|---------|
| K-Means | Elbow Method, Silhouette Score |
| DBSCAN | Noise Detection, Cluster Shapes |
| Hierarchical Clustering | Dendrogram Analysis |

---

## 🛠️ Requirements

### Python Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost imbalanced-learn tensorflow
```

### Menjalankan Notebook
1. Clone repository:
   ```bash
   git clone https://github.com/mrafiasyifaa/Midterm-Machine-Learning-2025.git
   cd Midterm-Machine-Learning-2025
   ```

2. Buka Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Jalankan notebook sesuai urutan:
   - `midterm_transaction_data.ipynb` - Classification
   - `midterm_regresi.ipynb` - Regression
   - `midterm_clustering.ipynb` - Clustering

---

## 📈 Metodologi

### Pipeline ML yang Digunakan:
1. **Data Loading & Exploration**
2. **Data Preprocessing**
   - Handling missing values
   - Feature engineering
   - Outlier detection
   - Feature scaling/normalization
3. **Exploratory Data Analysis (EDA)**
   - Visualization
   - Correlation analysis
   - Distribution analysis
4. **Model Training**
   - Multiple algorithms
   - Train-test split
   - Cross-validation
5. **Model Evaluation**
   - Metrics comparison
   - Confusion matrix (classification)
   - Residual plots (regression)
   - Cluster visualization (clustering)
6. **Model Comparison & Selection**

---

## 📝 Catatan
- Setiap notebook dilengkapi dengan penjelasan detail dan komentar
- Visualisasi disertakan untuk pemahaman yang lebih baik
- Model dibandingkan menggunakan metrics yang sesuai
- Hasil dapat direproduksi (random seed telah di-set)

---

## 🤝 Acknowledgments
- **Course**: Machine Learning and Deep Learning
- **Assignment**: UTS (Midterm) - Individual Task
- **Deadline**: 6 Desember 2025

---

**Last Updated**: 5 Desember 2025