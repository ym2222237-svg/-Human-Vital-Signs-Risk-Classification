# Human Vital Signs Risk Classification

## Overview

A machine learning application for classifying human vital signs and assessing health risk levels.

The project includes data preprocessing, exploratory data analysis, machine learning model training, model evaluation, and an interactive application for making predictions based on vital sign inputs.

---

## Objective

The objective of this project is to develop a machine learning solution that can classify health risk levels based on human vital signs.

The model uses the following input features:

- Heart Rate
- Respiratory Rate
- SpO2
- Temperature

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Preparation
5. Train/Test Split
6. Model Training
7. Model Evaluation
8. Model Selection
9. Model Deployment

---

## Models Evaluated

Three machine learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 96.87% | 99.17% | 96.75% | 97.95% | 99.68% |
| Random Forest | 99.71% | 99.82% | 99.80% | 99.81% | 99.99% |
| XGBoost | 99.49% | 99.97% | 99.37% | 99.67% | 99.99% |

---

## Final Model

The **Random Forest** model was selected and saved for deployment in the application.

The saved model is used by the application to generate predictions from user-provided vital sign values.

---

## Application

The project includes an interactive application that allows users to enter human vital sign measurements and receive a predicted risk classification.

### Input Features

The application uses:

- Heart Rate
- Respiratory Rate
- SpO2
- Temperature

### Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
