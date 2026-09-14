<div align="center">

# 🏦 Customer Churn Prediction with ANN

### An end-to-end Artificial Neural Network project for predicting bank customer churn

Built with **TensorFlow/Keras**, **Scikit-learn**, **Pandas**, **NumPy**, and **Streamlit**.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
</p>

</div>

---

## 📌 Overview

Customer churn is a practical binary classification problem: given a customer's profile and banking activity, can we estimate the likelihood that the customer will leave the bank?

This project builds an end-to-end **Customer Churn Prediction system** using an **Artificial Neural Network (ANN)** trained on the Churn Modelling dataset.

The trained model is integrated with a **Streamlit web application**, allowing users to enter customer information and receive a predicted churn probability along with a simple classification.

The project covers the complete workflow:

**Data → Preprocessing → Feature Engineering → ANN Training → Model Persistence → Prediction → Deployment**

---

## ✨ Key Features

* 🧠 Artificial Neural Network built with TensorFlow/Keras
* 📊 Binary classification for customer churn
* 🔤 Label encoding for categorical variables
* 🌍 One-hot encoding for geography
* 📏 Feature standardization using `StandardScaler`
* 💾 Saved model and preprocessing artifacts
* 🖥️ Interactive Streamlit prediction interface
* 📓 Jupyter notebooks for experimentation and training
* 📈 TensorBoard support for training logs

---

## 🎯 Prediction Target

The model predicts the `Exited` column from the Churn Modelling dataset.

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Customer did not churn |
| `1`   | Customer churned       |

The Streamlit application uses a **0.50 probability threshold**:

```text
Probability > 0.50
        ↓
Likely to churn

Probability ≤ 0.50
        ↓
May not churn
```

> **Note:** The predicted probability represents the model's estimate and should not be interpreted as a guarantee of customer behavior.

---

## 🧾 Model Inputs

The application accepts the following customer attributes:

| Feature           | Description                              |
| ----------------- | ---------------------------------------- |
| `Geography`       | Customer's country/region                |
| `Gender`          | Customer gender                          |
| `Age`             | Customer age                             |
| `CreditScore`     | Customer credit score                    |
| `Balance`         | Customer account balance                 |
| `EstimatedSalary` | Estimated customer salary                |
| `Tenure`          | Number of years with the bank            |
| `NumOfProducts`   | Number of bank products used             |
| `HasCrCard`       | Whether the customer has a credit card   |
| `IsActiveMember`  | Whether the customer is an active member |

During training, the following columns are excluded because they are identifiers rather than useful predictive features:

* `RowNumber`
* `CustomerId`
* `Surname`

The target variable is:

```text
Exited
```

---

## 🧠 Machine Learning Pipeline

```text
                Churn Modelling Dataset
                         │
                         ▼
                Data Cleaning
                         │
                         ▼
                Feature Selection
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Gender Encoding       Geography Encoding
       LabelEncoder          OneHotEncoder
              │                     │
              └──────────┬──────────┘
                         ▼
                  Feature Scaling
                  StandardScaler
                         │
                         ▼
             Artificial Neural Network
                 TensorFlow / Keras
                         │
                         ▼
                Churn Probability
                         │
                         ▼
                  0.50 Threshold
                    /         \
                   /           \
                  ▼             ▼
             Likely to       May not
              churn           churn
```

---

## 🏗️ Architecture

The project follows a simple and practical machine learning architecture:

```text
                    ┌─────────────────────┐
                    │   Customer Inputs   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Preprocessing    │
                    │                     │
                    │ Label Encoding      │
                    │ One-Hot Encoding    │
                    │ Standard Scaling    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ANN Model       │
                    │  TensorFlow/Keras   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Churn Probability   │
                    │      0 → 1          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └─────────────────────┘
```

---

## 📂 Project Structure

```text
ANN-Binary-Classification/
│
├── app.py
├── Churn_Modelling.csv
│
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
│
├── experiments.ipynb
├── predictions.ipynb
├── test.ipynb
│
├── requirements.txt
│
└── logs/
```

### File Description

| File                       | Purpose                                           |
| -------------------------- | ------------------------------------------------- |
| `app.py`                   | Streamlit application for interactive predictions |
| `Churn_Modelling.csv`      | Customer churn dataset                            |
| `model.h5`                 | Trained TensorFlow/Keras ANN model                |
| `scaler.pkl`               | Fitted `StandardScaler`                           |
| `label_encoder_gender.pkl` | Gender label encoder                              |
| `onehot_encoder_geo.pkl`   | Geography one-hot encoder                         |
| `experiments.ipynb`        | Data preprocessing and model training             |
| `predictions.ipynb`        | Example prediction workflow                       |
| `test.ipynb`               | Additional experiments/testing                    |
| `requirements.txt`         | Python dependencies                               |
| `logs/`                    | TensorBoard training logs                         |

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/AyAnand117/ANN-Binary-Classification.git

cd ANN-Binary-Classification
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses:

* TensorFlow `2.21.0`
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TensorBoard
* Matplotlib
* IPykernel

---

# ▶️ Run the Application

Once the dependencies are installed, run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

You can then enter customer information such as:

```text
Geography
Gender
Age
Credit Score
Balance
Estimated Salary
Tenure
Number of Products
Credit Card Status
Active Member Status
```

The application processes the input using the saved encoders and scaler before passing it to the trained ANN model.

The final output includes:

```text
Churn Probability: 0.xx
```

followed by the corresponding churn classification.

---

# 🧪 Training Workflow

The main experimentation and training workflow is available in:

```text
experiments.ipynb
```

The overall workflow is:

```text
Load Dataset
      ↓
Explore Data
      ↓
Select Features
      ↓
Encode Categorical Variables
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Build ANN
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Save Model
      ↓
Save Preprocessing Artifacts
```

The saved artifacts are then reused by the Streamlit application.

---

# 🔄 Prediction Workflow

The application performs preprocessing before inference:

```text
User Input
    │
    ▼
Gender Label Encoding
    │
    ▼
Geography One-Hot Encoding
    │
    ▼
Feature Combination
    │
    ▼
StandardScaler
    │
    ▼
ANN Model
    │
    ▼
Prediction Probability
    │
    ▼
Classification
```

This ensures that new customer inputs are transformed in a way that is consistent with the training pipeline.

---

# 📊 Dataset

The project uses the **Churn Modelling** dataset.

The dataset contains customer information such as:

* Credit score
* Geography
* Gender
* Age
* Tenure
* Account balance
* Number of products
* Credit card ownership
* Active membership
* Estimated salary
* Churn status

The target variable is:

```text
Exited
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Machine Learning

* TensorFlow
* Keras
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization & Experimentation

* Matplotlib
* Jupyter Notebook
* TensorBoard

### Deployment

* Streamlit

---

# 🔍 Key Learning Objectives

This project demonstrates how to take a machine learning classification problem from experimentation to a usable application.

### Machine Learning

* Binary classification
* Artificial Neural Networks
* Model training
* Model evaluation
* Prediction probability

### Feature Engineering

* Label encoding
* One-hot encoding
* Feature scaling
* Consistent preprocessing between training and inference

### Model Deployment

* Saving trained models
* Loading preprocessing artifacts
* Building a Streamlit interface
* Performing real-time inference

---

# 💡 Why This Project?

A machine learning model becomes significantly more useful when it can be consumed outside a notebook.

This project focuses on that transition:

```text
                 ML Experiment
                      │
                      ▼
                Trained Model
                      │
                      ▼
             Saved Model Artifacts
                      │
                      ▼
              Prediction Pipeline
                      │
                      ▼
               Streamlit App
                      │
                      ▼
                 User Input
                      │
                      ▼
             Real-Time Prediction
```

It provides a practical example of connecting **machine learning development with application deployment**.

---

# ⚠️ Important Notes

* Run the application from the **repository root** so that the model and pickle files can be located correctly.
* The application currently uses a fixed `0.50` classification threshold.
* The model outputs a probability between `0` and `1`.
* The project includes pre-trained model and preprocessing artifacts, so retraining is not required to run the application.
* TensorBoard logs may be generated under `logs/fit/` during training.
* This project is intended as a machine learning demonstration and should not be considered a production banking decision system without additional validation, monitoring, calibration, security, and fairness analysis.

---

# 🚧 Future Improvements

Some potential improvements include:

* [ ] Add ROC-AUC, precision, recall, and F1-score
* [ ] Add confusion matrix visualization
* [ ] Add model performance dashboard
* [ ] Perform hyperparameter tuning
* [ ] Optimize the classification threshold
* [ ] Add stronger input validation
* [ ] Add automated unit tests
* [ ] Add model versioning
* [ ] Containerize the application with Docker
* [ ] Deploy the application to a cloud platform
* [ ] Add model monitoring
* [ ] Add explainability using SHAP or similar techniques

---

# 📜 License

No separate license is currently specified for this repository.

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star!

### Built with Python, TensorFlow, Scikit-learn & Streamlit

</div>
