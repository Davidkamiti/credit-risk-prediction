# Credit-risk-prediction
## Overview

This project aims to develop a machine learning model to predict whether a loan applicant will default on their loan based on various financial and loan-related features. The model is implemented using a Random Forest classifier, which has been fine-tuned for optimal performance. The project also includes a deployment script using Streamlit, which provides an interactive web interface for making predictions.

## Dataset

The dataset used for training the model includes the following features:

#### person_income:
Annual income of the loan applicant.

#### loan_amnt:
Amount of the loan requested.

#### loan_grade:
Grade assigned to the loan by the lender.

loan_intent: The intended use of the loan (e.g., personal, education, medical).

loan_int_rate: Interest rate of the loan.

person_home_ownership: Home ownership status of the applicant (e.g., rent, own, mortgage).

person_emp_length: Length of employment in years.

person_age: Age of the loan applicant.

cb_person_cred_hist_length: Length of the applicant's credit history.

cb_person_default_on_file: Whether the applicant has any defaults on file (yes/no).

## Model Development

The model development process involved the following steps:

1. Data Preprocessing: Cleaning the data, handling missing values, and encoding categorical variables.
2. Model Training: Training a Random Forest classifier on the preprocessed data.
3. Hyperparameter Tuning: Using GridSearchCV to find the best hyperparameters for the model.
4. Model Evaluation: Evaluating the model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation was also performed to ensure the model's robustness.
## Results
After hyperparameter tuning, the model achieved the following performance metrics:

Accuracy: 90%
Precision: 92% for class 0 (no default) and 81% for class 1 (default)
Recall: 95% for class 0 and 71% for class 1
F1-Score: 94% for class 0 and 76% for class 1
ROC-AUC: 92%
## Model Deployment

The model is deployed using Streamlit, a Python library that allows for the creation of web applications with ease. The deployment script app.py provides a user-friendly interface for making predictions based on user input.
