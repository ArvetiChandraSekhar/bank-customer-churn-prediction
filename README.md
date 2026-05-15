# 🏦 Bank Customer Churn Prediction

> A machine learning application that predicts whether bank customers are likely to churn, using demographic and behavioral data across 5 classification models.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Customer churn is one of the most critical challenges in retail banking. This project builds an end-to-end ML pipeline to:

- Identify customers at high risk of leaving the bank
- Understand which features most influence churn
- Compare multiple classification models to find the best predictor
- Handle real-world class imbalance using SMOTE

---

## Dataset

**Source:** [Kaggle — Bank Customer Churn Records](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country of residence (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Annual estimated salary |
| Card Type | Type of card held |
| Complain | Whether the customer has filed a complaint |
| Satisfaction Score | Customer satisfaction rating |
| **Exited** | **Target: 1 = churned, 0 = retained** |

**Class Distribution:** ~20% churn (class imbalance handled with SMOTE)

---

## Project Structure

```
bank-customer-churn-prediction/
│
├── bank-customer-churn.ipynb   # Main notebook (EDA + modeling)
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/ArvetiChandraSekhar/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up Kaggle API**

Download your `kaggle.json` API token from [kaggle.com/settings](https://www.kaggle.com/settings) and place it at `~/.kaggle/kaggle.json`.

---

## Usage

Open and run the notebook:

```bash
jupyter notebook bank-customer-churn.ipynb
```

Run all cells in order. The notebook will:
1. Download the dataset via `kagglehub`
2. Perform data cleaning and EDA
3. Train all 5 models
4. Print comparison metrics and confusion matrices

---

## Methodology

### 1. Data Cleaning
- Dropped non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
- Retained behavioral and demographic features

### 2. Exploratory Data Analysis
- Visualized target class distribution
- Computed feature-target correlation matrix and heatmap
- Ran **chi-square tests** on categorical features (`Geography`, `Gender`, `Card Type`) to confirm statistical significance

### 3. Feature Engineering
- Applied **one-hot encoding** to categorical variables (`pd.get_dummies`)
- Optional: drop leakage features (`Complain`, `Satisfaction Score`) via a configurable flag

### 4. Class Imbalance — SMOTE
The dataset has ~80% retained vs ~20% churned customers. To prevent models from ignoring the minority class:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### 5. Train/Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

---

## Models & Results

Five models were trained and evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline model |
| K-Nearest Neighbors (KNN) | Instance-based learning (k=5) |
| Random Forest | Ensemble of 100 decision trees |
| SVM (RBF kernel) | Non-linear support vector classifier |
| SVM (Polynomial kernel) | Polynomial boundary support vector classifier |

**Evaluation Metrics:**

Each model is assessed on:
- **Accuracy** — Overall correctness
- **Precision** — Of predicted churners, how many actually churned
- **Recall** — Of actual churners, how many were caught *(most important for churn)*
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the receiver operating characteristic curve
- **Confusion Matrix** — Visual breakdown of TP, TN, FP, FN

> ⚠️ **Why Recall matters most:** Missing a churner (false negative) is far more costly than a false alarm. Retention campaigns are cheap; losing a customer is expensive.

---

## Key Insights

- **Age** is one of the strongest predictors of churn — older customers tend to churn more
- **Active membership** is negatively correlated with churn — engaged customers stay
- **Geography** is statistically significant — German customers show notably higher churn rates
- **SMOTE dramatically improved recall** — without it, models defaulted to predicting "no churn" for almost everyone
- **Random Forest** was the top performer overall on F1 and ROC-AUC

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | ML models, metrics, preprocessing |
| `imbalanced-learn` | SMOTE oversampling |
| `scipy` | Chi-square statistical tests |
| `kagglehub` | Dataset download |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
kagglehub
jupyter
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

 [Chandra Sekhar](https://github.com/ArvetiChandraSekhar)

