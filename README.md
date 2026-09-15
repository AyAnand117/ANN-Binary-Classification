# 🧠 ANN Machine Learning Projects

A practical collection of **Artificial Neural Network (ANN)** projects built with **TensorFlow/Keras**, **Scikit-learn**, **Pandas**, **NumPy**, and **Streamlit**.

This repository demonstrates how the same customer dataset can be used for different supervised learning problems:

* **Binary classification** for customer churn prediction
* **Regression** for estimated salary prediction
* **Hyperparameter tuning** for ANN classification
* **Model inference through Streamlit applications**
* **Training monitoring with TensorBoard**

> **Note:** Although the repository is named `ANN-Binary-Classification`, it has evolved beyond classification and now includes regression and hyperparameter-tuning experiments.

---

## 📌 Project Overview

The repository uses the `Churn_Modelling.csv` customer dataset and builds separate ANN workflows around different target variables.

```text
                         Customer Dataset
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
          Classification   Hyperparameter   Regression
             Churn           Tuning        Salary Prediction
                │               │               │
                ▼               ▼               ▼
             ANN Model       Grid Search      ANN Model
                │                               │
                └───────────────┬───────────────┘
                                ▼
                       Saved Model + Artifacts
                                │
                                ▼
                         Streamlit Apps
```

The main focus is the complete workflow from **data preprocessing and experimentation to model training, persistence, and interactive inference**.

---

# 🚀 Projects

## 1. Customer Churn Prediction

A binary classification project that predicts whether a customer is likely to leave the bank.

### Target

```text
Exited
```

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Customer did not churn |
| `1`   | Customer churned       |

### Input Features

The Streamlit application accepts:

* Geography
* Gender
* Age
* Credit Score
* Balance
* Estimated Salary
* Tenure
* Number of Products
* Credit Card Status
* Active Member Status

### Prediction

The ANN produces a churn probability.

The current application uses a `0.50` decision threshold:

```text
Probability > 0.50  →  Likely to churn
Probability ≤ 0.50  →  May not churn
```

### Key Files

* `experiments.ipynb` → classification experimentation and model development
* `predictions.ipynb` → prediction-related experiments
* `model.h5` → trained classification model
* `app.py` → Streamlit inference application
* `scaler.pkl` → fitted feature scaler
* `label_encoder_gender.pkl` → fitted gender encoder
* `onehot_encoder_geo.pkl` → fitted geography encoder

### Run the Application

```bash
streamlit run app.py
```

---

# 2. 💰 Estimated Salary Prediction

The repository also contains an ANN-based **regression workflow** that predicts estimated salary.

The same customer dataset is used, but the target variable is changed from `Exited` to:

```text
EstimatedSalary
```

Because salary is a continuous numerical value, this becomes a **regression problem**.

### Input Features

The regression application accepts:

* Geography
* Gender
* Age
* Balance
* Credit Score
* Exited
* Tenure
* Number of Products
* Credit Card Status
* Active Member Status

### Key Files

* `salaryregression.ipynb` → regression experimentation and training
* `regression_model.keras` → trained regression model
* `regression_app.py` → Streamlit inference application
* `scaler_reg.pkl` → regression feature scaler
* `label_encoder_gender_reg.pkl` → regression gender encoder
* `onehot_encoder_geo_reg.pkl` → regression geography encoder

### Run the Application

```bash
streamlit run regression_app.py
```

The application loads the trained model and its preprocessing artifacts before generating a salary prediction.

---

# 🔬 Hyperparameter Tuning

The repository includes a dedicated ANN hyperparameter-tuning experiment:

```text
hyperparametertuning.ipynb
```

The notebook integrates:

* `SciKeras`
* `KerasClassifier`
* `GridSearchCV`
* TensorFlow/Keras

### Parameters Explored

| Parameter        | Values                  |
| ---------------- | ----------------------- |
| Neurons          | `16`, `32`, `64`, `128` |
| Layers           | `1`, `2`                |
| Epochs           | `50`, `100`             |
| Cross-validation | `3-fold`                |

The saved notebook experiment reported the following best configuration:

```text
Neurons = 32
Layers  = 1
Epochs  = 50
```

with a cross-validation score of approximately:

```text
0.857
```

This is the result of the recorded notebook experiment and should not be treated as a production benchmark.

The hyperparameter-tuning preprocessing artifacts are:

```text
label_encoder_gender_hpt.pkl
onehot_encode_geo_hpt.pkl
scaler_hpt.pkl
```

---

# 🧩 Data Preprocessing

The project uses Scikit-learn preprocessing techniques before feeding the data into the ANN models.

## 1. Remove Identifier Columns

The notebooks remove:

```text
RowNumber
CustomerId
Surname
```

These fields act as identifiers rather than meaningful predictive features for the demonstrated workflows.

## 2. Gender Encoding

Gender is transformed using:

```python
LabelEncoder()
```

## 3. Geography Encoding

Geography is transformed using:

```python
OneHotEncoder()
```

producing features such as:

```text
Geography_France
Geography_Germany
Geography_Spain
```

## 4. Feature Scaling

Numerical inputs are standardized using:

```python
StandardScaler()
```

The fitted preprocessing objects are persisted as `.pkl` files.

This allows the Streamlit applications to apply the **same transformations used during model training** before generating predictions.

---

# 🧠 ANN Architecture

The models are built using TensorFlow/Keras dense neural-network layers.

## Classification

```text
Input Features
      ↓
Dense Layers
      ↓
ReLU Activations
      ↓
Output Layer
      ↓
Sigmoid
      ↓
Churn Probability
```

## Regression

```text
Input Features
      ↓
Dense Layers
      ↓
ReLU Activations
      ↓
Output Layer
      ↓
Continuous Prediction
      ↓
Estimated Salary
```

The hyperparameter-tuning workflow dynamically changes the number of neurons and dense layers to compare different ANN configurations.

---

# 🔄 End-to-End Machine Learning Workflow

```text
                Raw Dataset
                     │
                     ▼
               Data Cleaning
                     │
                     ▼
            Feature Engineering
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
     Categorical Data     Numerical Data
           │                   │
           ▼                   ▼
   Label / One-Hot        StandardScaler
      Encoding
           │                   │
           └─────────┬─────────┘
                     ▼
                Train / Test
                     │
                     ▼
                ANN Training
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Classification          Regression
          │                     │
          ▼                     ▼
    Saved Model            Saved Model
          │                     │
          └──────────┬──────────┘
                     ▼
            Serialized Artifacts
                     │
                     ▼
             Streamlit Inference
```

This demonstrates how a machine learning project can move from experimentation in notebooks to a reusable prediction application.

---

# 📈 TensorBoard & Experiment Tracking

Training logs are maintained separately for the classification and regression workflows:

```text
logs/
regressionlogs/
```

### Classification

```bash
tensorboard --logdir logs
```

### Regression

```bash
tensorboard --logdir regressionlogs
```

The repository also contains model and training visualizations inside:

```text
screenshots/
```

---

# 📂 Repository Structure

```text
ANN-Binary-Classification/
│
├── Churn_Modelling.csv
├── LICENSE
├── README.md
├── requirements.txt
│
├── app.py
├── experiments.ipynb
├── predictions.ipynb
├── hyperparametertuning.ipynb
├── model.h5
│
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scaler.pkl
│
├── label_encoder_gender_hpt.pkl
├── onehot_encode_geo_hpt.pkl
├── scaler_hpt.pkl
│
├── regression_app.py
├── salaryregression.ipynb
├── regression_model.keras
│
├── label_encoder_gender_reg.pkl
├── onehot_encoder_geo_reg.pkl
├── scaler_reg.pkl
│
├── logs/
├── regressionlogs/
└── screenshots/
```

---

# 📁 File Guide

| File                         | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `Churn_Modelling.csv`        | Customer dataset used by the experiments |
| `experiments.ipynb`          | ANN classification workflow              |
| `predictions.ipynb`          | Classification prediction experiments    |
| `hyperparametertuning.ipynb` | ANN hyperparameter search                |
| `app.py`                     | Streamlit churn prediction application   |
| `model.h5`                   | Saved ANN classification model           |
| `salaryregression.ipynb`     | ANN salary regression workflow           |
| `regression_app.py`          | Streamlit salary prediction application  |
| `regression_model.keras`     | Saved ANN regression model               |
| `*_gender*.pkl`              | Serialized gender encoders               |
| `onehot_*geo*.pkl`           | Serialized geography encoders            |
| `scaler*.pkl`                | Serialized feature scalers               |
| `logs/`                      | Classification TensorBoard logs          |
| `regressionlogs/`            | Regression TensorBoard logs              |
| `screenshots/`               | Training and model visualizations        |
| `requirements.txt`           | Python dependencies                      |
| `LICENSE`                    | GNU GPL v3 license                       |

---

# 🛠️ Tech Stack

| Category             | Technologies           |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Deep Learning        | TensorFlow, Keras      |
| Machine Learning     | Scikit-learn, SciKeras |
| Data Processing      | Pandas, NumPy          |
| Visualization        | Matplotlib             |
| Experiment Tracking  | TensorBoard            |
| Deployment           | Streamlit              |
| Notebook Environment | Jupyter / IPykernel    |

The current `requirements.txt` includes TensorFlow `2.21.0`, Pandas, NumPy, Scikit-learn, TensorBoard, Matplotlib, Streamlit, IPykernel, Keras, and SciKeras.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <repository-url>
cd ANN-Binary-Classification
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Customer Churn Prediction

```bash
streamlit run app.py
```

## 5. Run Salary Prediction

```bash
streamlit run regression_app.py
```

Run the applications from the repository root because the current inference scripts load their saved models and preprocessing artifacts using repository-relative paths.

---

# 🧪 Working With the Notebooks

## Classification

Start with:

```text
experiments.ipynb
```

This contains the main classification experimentation workflow.

For prediction-related experimentation:

```text
predictions.ipynb
```

## Regression

For salary prediction:

```text
salaryregression.ipynb
```

This workflow uses `EstimatedSalary` as the target variable.

## Hyperparameter Tuning

For ANN architecture experimentation:

```text
hyperparametertuning.ipynb
```

This notebook uses SciKeras and GridSearchCV to evaluate different combinations of ANN layers, neurons, and epochs.

---

# 🎯 Learning Objectives

This repository provides practical experience with:

### Machine Learning

* Binary classification
* Regression
* Train/test splitting
* Model evaluation
* Hyperparameter tuning

### Deep Learning

* Artificial Neural Networks
* Dense layers
* Activation functions
* Sigmoid output for binary classification
* ANN-based regression
* TensorFlow/Keras model training

### Data Preprocessing

* Feature selection
* Label encoding
* One-hot encoding
* Feature scaling
* Reusing fitted preprocessing objects

### Experimentation

* GridSearchCV
* SciKeras
* Cross-validation
* TensorBoard
* Training visualization

### Deployment

* Streamlit
* Model loading
* Preprocessing during inference
* Interactive prediction applications

---

# 💡 What This Project Demonstrates

The goal is not simply to train an ANN.

The repository demonstrates the progression from:

```text
Dataset
   ↓
Exploration
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
ANN Development
   ↓
Hyperparameter Tuning
   ↓
Model Training
   ↓
Model Persistence
   ↓
Inference Pipeline
   ↓
Streamlit Application
```

This provides a practical example of taking machine learning experimentation from a notebook into a usable application.

---

# ⚠️ Notes & Limitations

This repository is primarily intended for **learning, experimentation, and portfolio demonstration**.

The models should not be considered production-ready banking or financial decision systems without additional work involving:

* Robust validation
* Model calibration
* Fairness and bias analysis
* Security
* Input validation
* Monitoring
* Retraining strategies
* Domain-specific evaluation

Similarly, the salary prediction model demonstrates an ANN regression workflow using the provided dataset and should not be interpreted as a real-world salary estimation system.

---

# 🚧 Future Improvements

Potential next steps include:

* [ ] Add verified classification metrics
* [ ] Add confusion matrix
* [ ] Add ROC-AUC and Precision-Recall curves
* [ ] Add regression metrics such as MAE, MSE, RMSE, and R²
* [ ] Improve Streamlit input validation
* [ ] Improve Streamlit UI/UX
* [ ] Add model explainability
* [ ] Add automated tests
* [ ] Add experiment tracking/versioning
* [ ] Containerize applications with Docker
* [ ] Add CI/CD
* [ ] Add model monitoring
* [ ] Deploy applications to the cloud

---

# 📜 License

This project is licensed under the **GNU General Public License v3.0**.

See `LICENSE` for the complete license text.

---

# 👨‍💻 Author

**Ayush Anand**

This repository is part of an ongoing machine learning and AI engineering portfolio focused on building practical models and taking them from experimentation to usable applications.
