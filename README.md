# Telecom Churn Prediction

This project predicts customer churn in the telecom sector using supervised machine learning models. By analyzing customer demographics, service usage patterns, and billing details, the system identifies customers who are likely to discontinue service, enabling proactive retention strategies.

## 1. Project Overview

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Model Training (multiple ML algorithms)  
- Hyperparameter Tuning (RandomizedSearchCV)  
- Model Evaluation  
- Final Model Selection (CatBoost)

## 2. Dataset

- **Source:** Kaggle  
- **Records:** 50,000  
- **Features:** 14  
- **Target:** 1 → churn, 0 → stay  

## 3. Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction
```

### Step 2 — Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4 — Launch Jupyter Notebook
```bash
jupyter notebook
```

## 4. Required Packages

```
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
```

## 5. Data Cleaning & Preprocessing

- Removed duplicates  
- Encoded categorical variables  
- Standardized numerical values  
- Train-test split (80:20)  
- Checked class imbalance  

## 6. Exploratory Data Analysis (EDA)

- Churn distribution  
- Histograms  
- Correlation heatmap  
- Outlier detection  

## 7. Machine Learning Models

- Logistic Regression  
- Decision Tree  
- Random Forest  
- SVM  
- XGBoost  
- LightGBM  
- CatBoost  

## 8. Hyperparameter Tuning

- RandomizedSearchCV  
- Cross-validation  
- Feature selection  

## 9. Final Model — CatBoost Classifier

- Accuracy: 87%  
- F1–score: 0.82  

## 10. Business Insights

- Month-to-month → highest churn  
- Electronic check users churn more  
- High monthly charges increase churn  
- Long-term contracts reduce churn  

## 11. Project Structure

```
Telecom-Churn-Prediction/
│── data/
│── notebooks/
│── src/
│── results/
│── abstract.pdf
│── README.md
│── requirements.txt
```

## 12. Future Enhancements

- Deployment (Flask/Streamlit)  
- SHAP explainability  
- Automated feature engineering  
- Dashboard for monitoring  

## 13. License

For educational and research use.
