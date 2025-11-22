# Telecom Churn Prediction

This project predicts customer churn in the telecom sector using supervised machine learning models. By analyzing customer demographics, service usage patterns, and billing details, the system identifies customers who are likely to discontinue service, enabling proactive retention strategies.

---

## 1. Project Overview

This project includes:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Model Training using multiple ML algorithms  
- Hyperparameter Tuning using RandomizedSearchCV  
- Model Evaluation  
- Final Model Selection (CatBoost)

---

## 2. Dataset

- **Source:** Kaggle (Telecom Customer Churn Dataset)  
- **Records:** 50,000 (after cleaning)  
- **Features:** 14  
- **Target Variable:**  
  - `1` → Customer churns  
  - `0` → Customer stays  

---

## 3. Installation & Setup

Follow the steps below to run this project on your system.

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction
Step 2 — Create a Virtual Environment
Windows

bash
Copy code
python -m venv venv
venv\Scripts\activate
macOS / Linux

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Step 3 — Install Required Packages
bash
Copy code
pip install -r requirements.txt
Step 4 — Launch Jupyter Notebook
bash
Copy code
jupyter notebook
Open the following files:

notebooks/DATA_ANALYSIS.ipynb

notebooks/test_final.ipynb

4. Required Packages
The major libraries used in this project include:

nginx
Copy code
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
jupyter
imblearn
All dependencies will be installed through requirements.txt.

5. Data Cleaning & Preprocessing
Removed duplicates and irrelevant identifiers

Encoded categorical variables using LabelEncoder

Standardized numerical values using StandardScaler

Applied an 80:20 train–test split

Explored class imbalance (SMOTE considered)

6. Exploratory Data Analysis (EDA)
EDA included:

Churn distribution visualization

Histograms: tenure, monthly charges, usage patterns

Correlation heatmap

Outlier analysis

Key Insights:

Low-tenure customers churn more

High monthly charges increase churn probability

Contract type and payment method strongly influence churn

7. Machine Learning Models
Models evaluated:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

XGBoost

LightGBM

CatBoost (Best Performer)

Evaluation Metrics:
Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

8. Hyperparameter Tuning
Model optimization included:

RandomizedSearchCV

Built-in cross-validation

Feature importance analysis

Key selected features:

Tenure

Contract Type

Monthly Charges

Payment Method

9. Final Model — CatBoost Classifier
Performance:

Accuracy: 87%

F1-score: 0.82

Minimal preprocessing required

Strong precision–recall balance

Excellent generalization performance

10. Business Insights
Customers on month-to-month contracts churn more frequently

Electronic check payment users show higher churn

High monthly charges drive churn

Long-term contracts improve retention
