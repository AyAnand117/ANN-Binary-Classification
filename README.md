# ANN-Based Customer Churn Classification & Salary Prediction

An end-to-end Artificial Neural Network (ANN) project built with **TensorFlow/Keras** to solve two customer analytics problems using the **Churn Modelling** dataset:

1. **Binary Classification:** predict whether a banking customer is likely to churn.
2. **Regression:** estimate a customer's salary from demographic and banking attributes.

The repository goes beyond model training. It includes **data preprocessing, categorical encoding, feature scaling, ANN development, hyperparameter tuning, TensorBoard experiment tracking, model persistence, inference workflows, and Streamlit applications**.

> **Recruiter Snapshot:** This project demonstrates practical exposure to the complete machine learning lifecycle, from raw tabular data and preprocessing to ANN development, systematic model selection, experiment tracking, artifact persistence, and interactive model deployment.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Highlights](#key-highlights)
* [Business Problem](#business-problem)
* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Machine Learning Workflow](#machine-learning-workflow)
* [1. Customer Churn Classification](#1-customer-churn-classification)
* [2. Hyperparameter Tuning](#2-hyperparameter-tuning)
* [3. Prediction and Inference](#3-prediction-and-inference)
* [4. Streamlit Classification App](#4-streamlit-classification-app)
* [5. Tuned Streamlit Dashboard](#5-tuned-streamlit-dashboard)
* [6. Salary Regression](#6-salary-regression)
* [Model Artifacts](#model-artifacts)
* [Evaluation Metrics](#evaluation-metrics)
* [TensorBoard Experiment Tracking](#tensorboard-experiment-tracking)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Running the Applications](#running-the-applications)
* [Reproducing the Experiments](#reproducing-the-experiments)
* [Engineering Practices Demonstrated](#engineering-practices-demonstrated)
* [Limitations and Future Improvements](#limitations-and-future-improvements)
* [Why This Project Matters](#why-this-project-matters)
* [License](#license)
* [Author](#author)

---

## Project Overview

This project explores how Artificial Neural Networks can be applied to structured customer data for both **classification** and **regression** tasks.

The primary use case is **customer churn prediction**.

Given customer information such as:

* Credit score
* Geography
* Gender
* Age
* Tenure
* Account balance
* Number of products
* Credit card ownership
* Active membership status
* Estimated salary

the classification model predicts the probability that a customer will leave the institution.

The repository also includes a separate regression pipeline where **EstimatedSalary** is treated as a continuous target.

The overall project demonstrates:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Categorical Encoding
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
ANN Development
     ↓
Model Training
     ↓
Validation
     ↓
Hyperparameter Search
     ↓
Model Selection
     ↓
Model Persistence
     ↓
Inference
     ↓
Streamlit Application
```

---

## Key Highlights

| Area                        | Implementation                     |
| --------------------------- | ---------------------------------- |
| Problem Types               | Binary Classification + Regression |
| Dataset                     | Churn Modelling Dataset            |
| Records                     | 10,000                             |
| Classification Model        | TensorFlow / Keras ANN             |
| Regression Model            | TensorFlow / Keras ANN             |
| Categorical Encoding        | LabelEncoder + OneHotEncoder       |
| Feature Scaling             | StandardScaler                     |
| Hyperparameter Optimization | GridSearchCV + SciKeras            |
| Cross Validation            | 3-Fold                             |
| Experiment Tracking         | TensorBoard                        |
| Model Deployment            | Streamlit                          |
| Model Persistence           | `.h5` / `.keras`                   |
| Preprocessing Persistence   | `.pkl`                             |
| Programming Language        | Python                             |

---

# Business Problem

Customer churn is an important business problem for banks and subscription-based businesses.

When customers leave, organizations lose revenue and potentially valuable long-term relationships.

A churn prediction system can help identify customers who may be at higher risk so that retention teams can prioritize them for proactive actions.

This project approaches the problem as a **binary classification task**:

```text
0 → Customer did not leave
1 → Customer left
```

The ANN produces a probability between `0` and `1`, which can then be converted into a binary prediction using a threshold.

---

# Dataset

The project uses:

```text
data/Churn_Modelling.csv
```

The dataset contains **10,000 customer records**.

It includes demographic, financial and account-related information.

## Classification Features

| Feature           | Description                              |
| ----------------- | ---------------------------------------- |
| `CreditScore`     | Customer credit score                    |
| `Geography`       | Customer location                        |
| `Gender`          | Customer gender                          |
| `Age`             | Customer age                             |
| `Tenure`          | Number of years with the institution     |
| `Balance`         | Account balance                          |
| `NumOfProducts`   | Number of products owned                 |
| `HasCrCard`       | Whether the customer has a credit card   |
| `IsActiveMember`  | Whether the customer is an active member |
| `EstimatedSalary` | Estimated salary                         |
| `Exited`          | Target variable for churn classification |

The following fields are removed during preprocessing:

```text
RowNumber
CustomerId
Surname
```

These fields are treated as identifiers rather than predictive features.

---

# Repository Structure

```text
ANN-Binary-Classification/
│
├── data/
│   └── Churn_Modelling.csv
│
├── classification/
│   ├── app.py
│   ├── experiments.ipynb
│   ├── predictions.ipynb
│   ├── model.h5
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   ├── scaler.pkl
│   └── logs/
│       └── fit/
│           └── <TensorBoard run data>
│
├── hyperparametertuning/
│   ├── hyperparametertuning.ipynb
│   ├── hpt_app.py
│   ├── optimal_model.h5
│   ├── label_encoder_gender_hpt.pkl
│   ├── onehot_encode_geo_hpt.pkl
│   ├── scaler_hpt.pkl
│   └── hptlogs/
│       └── grid_search/
│           └── <TensorBoard run data>
│
├── regression/
│   ├── salaryregression.ipynb
│   ├── regression_app.py
│   ├── regression_model.keras
│   ├── label_encoder_gender_reg.pkl
│   ├── onehot_encoder_geo_reg.pkl
│   ├── scaler_reg.pkl
│   └── regressionlogs/
│       └── fit/
│           └── <TensorBoard run data>
│
├── requirements.txt
└── LICENSE
```

---

# Machine Learning Workflow

The core classification workflow can be summarized as:

```text
Churn_Modelling.csv
        ↓
Remove identifier columns
        ↓
Encode categorical variables
        ↓
Separate X and y
        ↓
Train/Test Split
        ↓
StandardScaler
        ↓
ANN
        ↓
Validation
        ↓
Hyperparameter Tuning
        ↓
Optimized Model
        ↓
Save Model + Preprocessing Artifacts
        ↓
Prediction
```

---

# 1. Customer Churn Classification

## Data Preprocessing

### Removing Irrelevant Features

The project removes:

```python
data = data.drop(
    ['RowNumber', 'CustomerId', 'Surname'],
    axis=1
)
```

These identifiers are not required for the model to learn customer behavior.

---

## Categorical Encoding

### Gender

`Gender` is transformed using `LabelEncoder`.

```python
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
```

### Geography

`Geography` is transformed using `OneHotEncoder`.

This generates:

```text
Geography_France
Geography_Germany
Geography_Spain
```

The categorical encoders are persisted using pickle so the same transformation can be applied during inference.

---

## Train/Test Split

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates:

```text
Training Samples: 8,000
Testing Samples:  2,000
```

---

## Feature Scaling

A `StandardScaler` is used to standardize the model inputs.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The fitted scaler is saved and reused during prediction.

This helps maintain consistency between:

```text
Training data
      ↓
Model input
      ↓
Future inference data
```

---

# Baseline ANN Architecture

The initial classification experiment uses the following architecture:

```text
Input Layer
    ↓
Dense Layer: 64 neurons
    Activation: ReLU
    ↓
Dense Layer: 32 neurons
    Activation: ReLU
    ↓
Output Layer: 1 neuron
    Activation: Sigmoid
    ↓
Churn Probability
```

### Architecture

```python
model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),
    Dense(
        32,
        activation='relu'
    ),
    Dense(
        1,
        activation='sigmoid'
    )
])
```

The notebook reports:

```text
Total Parameters: 2,945
Trainable Parameters: 2,945
```

---

## Training Configuration

The baseline ANN uses:

| Configuration         | Value               |
| --------------------- | ------------------- |
| Optimizer             | Adam                |
| Loss Function         | Binary Crossentropy |
| Output Activation     | Sigmoid             |
| Metric                | Accuracy            |
| Maximum Epochs        | 100                 |
| Early Stopping        | Enabled             |
| TensorBoard           | Enabled             |
| Validation Monitoring | `val_loss`          |

Early stopping is configured using validation loss:

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

---

# 2. Hyperparameter Tuning

The repository contains a dedicated hyperparameter optimization workflow:

```text
hyperparametertuning/hyperparametertuning.ipynb
```

The ANN is wrapped using:

```text
SciKeras KerasClassifier
```

which allows the neural network to work with:

```text
scikit-learn GridSearchCV
```

---

## Search Space

The project searches across:

```python
param_grid = {
    'neurons': [16, 32, 64, 128],
    'layers': [1, 2],
    'epochs': [50, 100]
}
```

This produces:

```text
4 neuron configurations
×
2 layer configurations
×
2 epoch configurations

= 16 total configurations
```

Each configuration is evaluated using **3-fold cross-validation**.

Therefore:

```text
16 configurations × 3 folds
= 48 model fits
```

---

## Best Configuration

The recorded grid-search result is:

| Parameter           |   Best Value |
| ------------------- | -----------: |
| Hidden Layers       |        **1** |
| Neurons             |       **16** |
| Epochs              |      **100** |
| Best CV Score       | **0.856499** |
| Approx. CV Accuracy |   **85.65%** |

The optimized model is saved as:

```text
hyperparametertuning/optimal_model.h5
```

The important takeaway is that the final model configuration was selected through systematic experimentation rather than only through manual architecture selection.

---

# Tuned Model Training

After the best configuration was identified, the project recreated the optimized architecture and trained it.

```python
optimal_model = create_model(
    neurons=16,
    layers=1
)
```

The model was then trained for:

```text
Epochs: 100
Batch Size: 10
```

TensorBoard logging was also enabled.

---

# 3. Prediction and Inference

The repository contains:

```text
classification/predictions.ipynb
```

which demonstrates how to use the saved model for inference.

The inference pipeline mirrors the training preprocessing pipeline.

```text
Raw Customer Information
        ↓
Gender Encoding
        ↓
Geography One-Hot Encoding
        ↓
Feature Combination
        ↓
StandardScaler
        ↓
ANN Model
        ↓
Churn Probability
        ↓
Binary Decision
```

---

## Prediction Output

The ANN uses a sigmoid output, producing a value between:

```text
0 and 1
```

This value is treated as the estimated churn probability.

The application uses:

```text
0.50
```

as the classification threshold.

Conceptually:

```python
if prediction_proba > 0.5:
    # Likely to churn
else:
    # May not churn
```

---

# 4. Streamlit Classification App

The classification model is exposed through:

```text
classification/app.py
```

The application accepts customer inputs such as:

* Geography
* Gender
* Age
* Balance
* Credit Score
* Estimated Salary
* Tenure
* Number of Products
* Credit Card Status
* Active Member Status

The application then:

```text
Input
  ↓
Encoding
  ↓
Scaling
  ↓
ANN Inference
  ↓
Churn Probability
  ↓
Prediction
```

This demonstrates how a trained machine learning model can be moved beyond a notebook and connected to an interactive user interface.

---

# 5. Tuned Streamlit Dashboard

The optimized model has its own Streamlit interface:

```text
hyperparametertuning/hpt_app.py
```

This application loads:

```text
optimal_model.h5
label_encoder_gender_hpt.pkl
onehot_encode_geo_hpt.pkl
scaler_hpt.pkl
```

### Dashboard Features

* Cached model and artifact loading
* Two-column input interface
* Input range validation
* Human-readable Yes/No controls
* Churn probability displayed as a percentage
* Optimized model specification display
* Interactive prediction button
* Local application shutdown control

The dashboard identifies the selected model configuration as:

```text
Hidden Layers : 1
Neurons       : 16
Epochs        : 100
CV Score      : ~85.65%
```

---

# 6. Salary Regression

The repository also contains a separate regression implementation.

Location:

```text
regression/
```

The regression notebook is:

```text
regression/salaryregression.ipynb
```

In this workflow:

```text
Target = EstimatedSalary
```

Instead of predicting a binary churn label, the ANN predicts a continuous numeric value.

---

## Regression Workflow

```text
Raw Customer Data
        ↓
Remove Identifier Columns
        ↓
Encode Categorical Variables
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
ANN Regression Model
        ↓
Salary Prediction
```

The trained model is saved as:

```text
regression/regression_model.keras
```

The corresponding Streamlit application is:

```text
regression/regression_app.py
```

The application collects customer attributes and returns an estimated salary.

---

# Model Artifacts

The repository stores both models and the preprocessing objects required for inference.

## Classification

```text
classification/model.h5
classification/label_encoder_gender.pkl
classification/onehot_encoder_geo.pkl
classification/scaler.pkl
```

## Hyperparameter-Tuned Classification

```text
hyperparametertuning/optimal_model.h5
hyperparametertuning/label_encoder_gender_hpt.pkl
hyperparametertuning/onehot_encode_geo_hpt.pkl
hyperparametertuning/scaler_hpt.pkl
```

## Regression

```text
regression/regression_model.keras
regression/label_encoder_gender_reg.pkl
regression/onehot_encoder_geo_reg.pkl
regression/scaler_reg.pkl
```

Persisting preprocessing artifacts is important because inference data must go through the same transformation logic as the training data.

---

# Evaluation Metrics

The repository contains several model metrics and model outputs.

## Metrics Present

| Metric                                   | Usage                                                    |
| ---------------------------------------- | -------------------------------------------------------- |
| **Accuracy**                             | Classification training metric                           |
| **Validation Accuracy** (`val_accuracy`) | Tracks validation performance during training            |
| **Binary Cross-Entropy Loss**            | Classification optimization objective                    |
| **Validation Loss** (`val_loss`)         | Monitors validation error                                |
| **Grid Search Best Score**               | Used for selecting the best hyperparameter configuration |
| **Churn Probability**                    | Model output used for customer-level prediction          |

---

## Recorded Classification Results

### Baseline Model

Architecture:

```text
64 → 32 → 1
```

Parameters:

```text
2,945
```

Recorded training log:

```text
Validation Accuracy ≈ 86.50%
Validation Loss ≈ 0.3377
```

These values appear around epoch 10 in the stored training run.

---

## Hyperparameter Search

The best recorded configuration is:

```text
Hidden Layers : 1
Neurons       : 16
Epochs        : 100
```

Best recorded cross-validation score:

```text
0.856499
```

or approximately:

```text
85.65%
```

---

## Final Tuned Model Training

The stored final training run reports at epoch 100:

| Metric              |      Value |
| ------------------- | ---------: |
| Training Accuracy   | **86.46%** |
| Validation Accuracy | **85.50%** |
| Training Loss       | **0.3246** |
| Validation Loss     | **0.3452** |

### Important Interpretation

The following values represent different evaluation contexts:

```text
85.65%
↓
Best 3-Fold Cross-Validation Score from GridSearchCV

85.50%
↓
Validation Accuracy at Epoch 100 of Final Tuned Model Training
```

These figures should not be treated as interchangeable.

---

## Metrics Not Currently Reported

The repository does **not** currently contain a dedicated reported evaluation for:

```text
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
Confusion Matrix
Calibration
MAE
MSE
RMSE
R²
```

These metrics should not be claimed as project results unless they are explicitly calculated in a future evaluation step.

---

# TensorBoard Experiment Tracking

TensorBoard is used to monitor model training.

Relevant directories include:

```text
classification/logs/fit/
hyperparametertuning/hptlogs/grid_search/
regression/regressionlogs/fit/
```

TensorBoard allows inspection of:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Training progression
* Experiment behavior

---

## Run TensorBoard for Classification

```bash
tensorboard --logdir classification/logs/fit/
```

## Run TensorBoard for Hyperparameter Experiments

```bash
tensorboard --logdir hyperparametertuning/hptlogs/grid_search/
```

---

# Tech Stack

## Programming Language

```text
Python
```

## Machine Learning

```text
TensorFlow 2.21.0
Keras
scikit-learn
SciKeras
```

## Data Processing

```text
Pandas
NumPy
```

## Feature Preprocessing

```text
StandardScaler
LabelEncoder
OneHotEncoder
```

## Experiment Tracking

```text
TensorBoard
```

## Visualization

```text
Matplotlib
```

## Application / Deployment

```text
Streamlit
```

Dependencies are defined in:

```text
requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/AyAnand117/ANN-Binary-Classification.git
cd ANN-Binary-Classification
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Applications

## 1. Customer Churn Prediction

Navigate to:

```bash
cd classification
```

Run:

```bash
streamlit run app.py
```

---

## 2. Hyperparameter-Tuned Churn Dashboard

Navigate to:

```bash
cd hyperparametertuning
```

Run:

```bash
streamlit run hpt_app.py
```

---

## 3. Salary Regression Application

Navigate to:

```bash
cd regression
```

Run:

```bash
streamlit run regression_app.py
```

> Run each application from its respective directory because the application code uses relative paths to load models and preprocessing artifacts.

---

# Reproducing the Experiments

## Classification Experiment

Open:

```text
classification/experiments.ipynb
```

The notebook covers:

```text
Dataset Loading
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Categorical Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
ANN Construction
      ↓
Model Compilation
      ↓
Training
      ↓
Early Stopping
      ↓
TensorBoard Logging
      ↓
Model Saving
```

---

## Prediction Notebook

Open:

```text
classification/predictions.ipynb
```

This notebook demonstrates how to:

```text
Load the trained model
        ↓
Load preprocessing artifacts
        ↓
Prepare a customer record
        ↓
Transform categorical variables
        ↓
Scale the input
        ↓
Generate churn probability
```

---

## Hyperparameter Tuning

Open:

```text
hyperparametertuning/hyperparametertuning.ipynb
```

The notebook covers:

```text
Dataset Preparation
        ↓
SciKeras KerasClassifier
        ↓
GridSearchCV
        ↓
3-Fold Cross Validation
        ↓
16 Hyperparameter Configurations
        ↓
48 Total Fits
        ↓
Best Model Selection
        ↓
Final Training
        ↓
TensorBoard Logging
        ↓
Model Saving
```

---

## Regression

Open:

```text
regression/salaryregression.ipynb
```

This notebook covers the salary prediction problem using an ANN regression workflow.

---

# Engineering Practices Demonstrated

## Reproducible Preprocessing

The project saves preprocessing objects such as:

```text
LabelEncoder
OneHotEncoder
StandardScaler
```

This allows inference to use the same transformations as training.

---

## Separation of Training and Inference

The repository separates:

```text
Experimentation
    ↓
Notebooks
```

from:

```text
Model Serving
    ↓
Streamlit Applications
```

This creates a clearer separation between experimentation and application usage.

---

## Systematic Hyperparameter Optimization

Instead of relying purely on manual experimentation, the project uses:

```text
GridSearchCV
+
SciKeras
+
3-Fold Cross Validation
```

to compare ANN configurations.

---

## Experiment Tracking

TensorBoard logs are stored alongside the project to inspect model training behavior.

---

## Model Persistence

The trained neural networks are saved and reused instead of retraining them each time the application is launched.

---

## Interactive Model Deployment

The Streamlit applications provide a simple way for users to interact with the trained models.

---

# Limitations and Future Improvements

The project provides a solid applied machine-learning workflow, but there are several areas that could make it more production-ready.

## 1. Add Comprehensive Evaluation

A dedicated evaluation script could calculate:

```text
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
Confusion Matrix
Calibration Metrics
```

For regression:

```text
MAE
MSE
RMSE
R²
```

---

## 2. Improve Data Validation

Input validation and schema checks could be centralized so that both training and inference use the same rules.

---

## 3. Create a Reusable Preprocessing Pipeline

The preprocessing currently appears across separate notebooks and applications.

A single reusable pipeline could reduce duplication and make maintenance easier.

---

## 4. Separate Validation and Test Data

A dedicated untouched test set could be maintained separately from model-selection workflows for a cleaner final evaluation.

---

## 5. Modernize Model Serialization

The classification models are currently stored in `.h5`.

For future iterations, the native Keras format could be preferred.

---

## 6. Add Automated Testing

Tests could be added for:

```text
Preprocessing
Feature ordering
Model input shape
Inference consistency
Application behavior
```

---

## 7. Containerize the Application

Docker could be introduced to make the deployment environment more reproducible.

---

## 8. Add Experiment and Model Versioning

Production workflows could benefit from explicit:

```text
Model Version
Dataset Version
Experiment ID
Training Configuration
Evaluation Results
```

---

# Why This Project Matters

From a recruiter or hiring-manager perspective, the project demonstrates more than the ability to create a neural network.

It demonstrates the ability to connect multiple components of an applied machine learning workflow:

```text
Business Problem
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
ANN Architecture Design
      ↓
Model Training
      ↓
Validation
      ↓
Hyperparameter Optimization
      ↓
Experiment Tracking
      ↓
Model Persistence
      ↓
Inference Pipeline
      ↓
Interactive Deployment
```

### Skills demonstrated through the repository

```text
Python
Machine Learning
Deep Learning
Artificial Neural Networks
TensorFlow
Keras
Scikit-learn
SciKeras
Pandas
NumPy
Feature Engineering
Data Preprocessing
Hyperparameter Tuning
Cross Validation
Model Evaluation
TensorBoard
Model Serialization
Streamlit
```

### Strongest Portfolio Signals

The strongest aspects of the project are the combination of:

**ANN model development**

*

**reproducible preprocessing**

*

**systematic hyperparameter tuning**

*

**experiment tracking with TensorBoard**

*

**saved model artifacts**

*

**interactive Streamlit inference**

This demonstrates an understanding of how a machine learning model moves from experimentation toward an actual usable application.

---

# License

This project is distributed under the license included in the repository.

---

# Author

**Ayush Anand**

GitHub:

https://github.com/AyAnand117

Repository:

https://github.com/AyAnand117/ANN-Binary-Classification
