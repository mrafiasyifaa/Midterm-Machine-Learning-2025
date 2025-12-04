# Midterm Machine Learning 2025 - End-to-End ML Pipeline

## 📋 Student Information
- **Name**: [Your Full Name]
- **NIM**: [Your Student ID]
- **Class**: [Your Class]

---

## 🎯 Project Overview
This repository contains comprehensive end-to-end Machine Learning implementations for the Midterm examination. The project demonstrates practical knowledge in data preprocessing, model development, training, evaluation, and comparison across multiple ML algorithms using a real-world **Transaction Fraud Detection Dataset**.

The project covers three major ML approaches:
1. **Classification** - Fraud Detection System
2. **Regression** - Transaction Amount Prediction
3. **Clustering** - Customer Segmentation Analysis

---

## 📊 Dataset Description
**Transaction Fraud Detection Dataset**
- **Source**: Credit card transaction data
- **Target Variable**: `isFraud` (Binary: 0 = Legitimate, 1 = Fraud)
- **Features**: 
  - Transaction details (TransactionDT, TransactionAmt, ProductCD)
  - Card information (card1-card6)
  - Address information (addr1, addr2)
  - Distance metrics (dist1, dist2)
  - Email domains (P_emaildomain, R_emaildomain)
  - Categorical features (C1-C14)
  - Time-based features (D1-D15)
  - Match features (M1-M9)
  - Versioned features (V1-V339)

---

## 🚀 Models Implemented

### 1. Classification Models (Fraud Detection)
Located in: `midterm_transaction_data.ipynb`

| Model | Description | Key Metrics |
|-------|-------------|-------------|
| **Logistic Regression** | Linear model for binary classification | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Random Forest Classifier** | Ensemble of decision trees | Feature importance, Accuracy, ROC-AUC |
| **XGBoost Classifier** | Gradient boosting algorithm | Precision-Recall curve, ROC-AUC |
| **Neural Network (MLP)** | Multi-layer perceptron | Loss curves, Accuracy, F1-Score |

**Evaluation Metrics**:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- Feature Importance Analysis

---

### 2. Regression Models (Transaction Amount Prediction)
Located in: `midterm_regresi.ipynb`

| Model | Description | Key Metrics |
|-------|-------------|-------------|
| **Linear Regression** | Basic linear relationship modeling | R² Score, MSE, MAE |
| **Ridge Regression** | L2 regularization | R² Score, RMSE |
| **Lasso Regression** | L1 regularization with feature selection | R² Score, MAE |
| **Random Forest Regressor** | Ensemble regression trees | R² Score, Feature importance |
| **Gradient Boosting Regressor** | Advanced boosting technique | R² Score, RMSE, MAE |

**Evaluation Metrics**:
- R² Score (Coefficient of Determination)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Residual plots
- Prediction vs Actual plots

---

### 3. Clustering Models (Customer Segmentation)
Located in: `midterm_clustering.ipynb`

| Model | Description | Key Analysis |
|-------|-------------|--------------|
| **K-Means Clustering** | Partition-based clustering | Elbow method, Silhouette score |
| **DBSCAN** | Density-based clustering | Noise detection, Cluster shapes |
| **Hierarchical Clustering** | Agglomerative clustering | Dendrogram analysis |

**Evaluation Metrics**:
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score
- Cluster visualization (PCA/t-SNE)
- Cluster profiling

---

## 📁 Repository Structure
```
Midterm-Machine-Learning-2025/
│
├── README.md                              # Project documentation
├── midterm_transaction_data.ipynb         # Classification: Fraud Detection
├── midterm_regresi.ipynb                  # Regression: Amount Prediction
├── midterm_clustering.ipynb               # Clustering: Customer Segmentation
│
├── datasets/
│   └── transaction/
│       ├── train_transaction.csv          # Training dataset
│       └── test_transaction.csv           # Test dataset
│
├── models/                                # Saved trained models (generated)
└── midterm_folder/                        # Additional resources
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost imbalanced-learn
pip install tensorflow  # or pytorch for neural networks
```

### Running the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/Midterm-Machine-Learning-2025.git
   cd Midterm-Machine-Learning-2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run notebooks in order:
   - Start with `midterm_transaction_data.ipynb` for classification
   - Then `midterm_regresi.ipynb` for regression
   - Finally `midterm_clustering.ipynb` for clustering

---

## 📈 Key Results Summary

### Classification Performance
- **Best Model**: [Model Name]
- **ROC-AUC Score**: [Score]
- **F1-Score**: [Score]
- **Recall**: [Score] (Important for fraud detection)

### Regression Performance
- **Best Model**: [Model Name]
- **R² Score**: [Score]
- **RMSE**: [Value]

### Clustering Insights
- **Optimal Clusters**: [Number]
- **Silhouette Score**: [Score]
- **Key Segments**: [Brief description]

---

## 🔍 Methodology

### 1. Data Preprocessing
- Missing value imputation
- Feature engineering
- Outlier detection and handling
- Feature scaling (StandardScaler/MinMaxScaler)
- Handling imbalanced data (SMOTE for fraud detection)

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis
- Correlation analysis
- Feature relationships
- Target variable analysis

### 3. Model Training
- Train-test split (80-20)
- Cross-validation (5-fold)
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Model comparison

### 4. Evaluation & Validation
- Multiple metrics evaluation
- Confusion matrix analysis
- Learning curves
- Feature importance analysis

---

## 📚 Learning Outcomes

Through this project, the following skills were developed:
- ✅ End-to-end ML pipeline development
- ✅ Data preprocessing and feature engineering
- ✅ Model selection and comparison
- ✅ Handling imbalanced datasets
- ✅ Hyperparameter tuning
- ✅ Model evaluation using multiple metrics
- ✅ Visualization and interpretation of results
- ✅ Documentation and code organization

---

## 🔗 Navigation Guide

| Notebook | Purpose | What You'll Find |
|----------|---------|------------------|
| `midterm_transaction_data.ipynb` | **Classification** | Fraud detection using Logistic Regression, Random Forest, XGBoost, and Neural Networks |
| `midterm_regresi.ipynb` | **Regression** | Transaction amount prediction using Linear, Ridge, Lasso, Random Forest, and Gradient Boosting |
| `midterm_clustering.ipynb` | **Clustering** | Customer segmentation using K-Means, DBSCAN, and Hierarchical Clustering |

**Recommended Order**: Classification → Regression → Clustering

---

## 📝 Notes
- All notebooks contain detailed explanations and comments
- Code follows PEP 8 style guidelines
- Visualizations included for better understanding
- Models are compared using appropriate metrics
- Results are reproducible (random seeds set)

---

## 🤝 Acknowledgments
- **Course**: Machine Learning and Deep Learning
- **Assignment**: Midterm - Individual Task
- **Deadline**: December 6, 2025
- **Instructor**: [Instructor Name]

---

## 📄 License
This project is created for educational purposes as part of the Machine Learning course midterm examination.

---

**Last Updated**: December 4, 2025