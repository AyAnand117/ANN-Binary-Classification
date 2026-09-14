<div align="center">

# 🧠 ANN Machine Learning Projects

### Artificial Neural Networks for Classification & Regression

A collection of end-to-end **Artificial Neural Network projects** built with **TensorFlow/Keras**, **Scikit-learn**, **Pandas**, **NumPy**, and **Streamlit**.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge" alt="License">
</p>

<p>
  <a href="https://github.com/AyAnand117/ANN-Binary-Classification">GitHub Repository</a>
</p>

</div>

---

# 📌 Overview

This repository started as an **Artificial Neural Network based customer churn classification project** and has now been expanded to include a second ANN-based **regression problem for estimated salary prediction**.

The goal of the repository is to explore how Artificial Neural Networks can be applied to different types of supervised learning problems while following a complete machine learning workflow.

```text
                    ANN Projects
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Classification             Regression
             │                       │
             ▼                       ▼
      Customer Churn          Salary Prediction
             │                       │
             ▼                       ▼
        ANN Model               ANN Model
             │                       │
             ▼                       ▼
       Streamlit App            Streamlit App
```

The repository therefore demonstrates both:

* **Binary Classification**
* **Regression**

with reusable preprocessing, trained models, notebooks, and Streamlit applications.

---

# 🚀 Projects

## 1. 🏦 Customer Churn Prediction

A binary classification problem that predicts whether a bank customer is likely to churn.

### Objective

Given information about a bank customer, predict the probability that the customer will leave the bank.

### Target

```text
Exited
```

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Customer did not churn |
| `1`   | Customer churned       |

### Model Output

The model produces a probability between `0` and `1`.

The Streamlit application uses a threshold of `0.50`:

```text
Probability > 0.50
        ↓
Likely to churn

Probability ≤ 0.50
        ↓
May not churn
```

### Input Features

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

### Application

The model is deployed through:

```text
app.py
```

Run it with:

```bash
streamlit run app.py
```

---

# 2. 💰 Estimated Salary Prediction

The repository now also contains an ANN-based **regression problem** for predicting a customer's estimated salary.

This problem uses the same underlying customer dataset but changes the prediction objective.

Instead of predicting:

```text
Exited
```

the regression model predicts:

```text
EstimatedSalary
```

Since `EstimatedSalary` is a continuous numerical value, the task becomes a **regression problem**.

### Objective

Given customer information, estimate the customer's salary using an Artificial Neural Network.

### Model Type

```text
Artificial Neural Network
        ↓
Regression
        ↓
Continuous Salary Prediction
```

### Input Features

The regression application uses:

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

### Application

The regression model is served through:

```text
regression_app.py
```

Run it with:

```bash
streamlit run regression_app.py
```

The application loads:

```text
regression_model.keras
```

along with the regression-specific preprocessing artifacts.

---

# 🧠 Why Two Problems?

One of the interesting aspects of this repository is that the same customer dataset can be used to demonstrate two fundamentally different machine learning tasks.

### Classification

```text
Customer Information
        ↓
ANN
        ↓
Churn Probability
        ↓
Churn / No Churn
```

### Regression

```text
Customer Information
        ↓
ANN
        ↓
Estimated Salary
        ↓
Continuous Numerical Value
```

This makes the repository useful for understanding the difference between **classification and regression using neural networks**.

---

# 🔄 Machine Learning Workflow

Both projects follow a similar end-to-end workflow.

```text
                    Dataset
                       │
                       ▼
               Data Exploration
                       │
                       ▼
              Feature Selection
                       │
                       ▼
             Feature Engineering
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     Categorical Data       Numerical Data
            │                     │
            ▼                     ▼
     Encoding Methods        Scaling
            │                     │
            └──────────┬──────────┘
                       ▼
                 Train / Test
                       │
                       ▼
                ANN Architecture
                       │
                       ▼
                  Model Training
                       │
                       ▼
                   Evaluation
                       │
                       ▼
                Model Persistence
                       │
                       ▼
                Streamlit App
                       │
                       ▼
                 New Prediction
```

---

# 🧠 Neural Network Workflow

The repository explores the construction and training of ANN models using TensorFlow/Keras.

The classification workflow includes:

```text
Input Features
      ↓
Dense Layers
      ↓
Activation Functions
      ↓
Output Layer
      ↓
Sigmoid Probability
```

For regression:

```text
Input Features
      ↓
Dense Layers
      ↓
Activation Functions
      ↓
Output Layer
      ↓
Continuous Prediction
```

---

# 🛠️ Feature Engineering

The projects use Scikit-learn preprocessing techniques before passing the data to the neural networks.

### Gender

Gender is transformed using:

```python
LabelEncoder
```

### Geography

Geography is transformed using:

```python
OneHotEncoder
```

producing features such as:

```text
Geography_France
Geography_Germany
Geography_Spain
```

### Numerical Features

Numerical features are standardized using:

```python
StandardScaler
```

The fitted preprocessing objects are saved and reused during inference.

This helps ensure that the data entering the deployed model follows the same transformation process used during training.

---

# 📂 Project Structure

```text
ANN-Binary-Classification/
│
├── 📊 Dataset
│   └── Churn_Modelling.csv
│
├── 🧠 Classification
│   ├── app.py
│   ├── experiments.ipynb
│   ├── predictions.ipynb
│   ├── model.h5
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
│
├── 💰 Regression
│   ├── regression_app.py
│   ├── salaryregression.ipynb
│   ├── regression_model.keras
│   ├── scaler_reg.pkl
│   ├── label_encoder_gender_reg.pkl
│   └── onehot_encoder_geo_reg.pkl
│
├── 📈 Training Logs
│   ├── logs/
│   └── regressionlogs/
│
├── 📸 Screenshots
│   └── screenshots/
│
├── 📦 Configuration
│   └── requirements.txt
│
├── 📜 LICENSE
│
└── 📖 README.md
```

---

# 📁 Important Files

| File                           | Purpose                                                 |
| ------------------------------ | ------------------------------------------------------- |
| `Churn_Modelling.csv`          | Customer dataset used for both projects                 |
| `experiments.ipynb`            | Classification data preparation and ANN experimentation |
| `predictions.ipynb`            | Example churn prediction workflow                       |
| `app.py`                       | Streamlit churn prediction application                  |
| `model.h5`                     | Trained ANN classification model                        |
| `scaler.pkl`                   | Classification feature scaler                           |
| `label_encoder_gender.pkl`     | Classification gender encoder                           |
| `onehot_encoder_geo.pkl`       | Classification geography encoder                        |
| `salaryregression.ipynb`       | Salary regression experimentation and training          |
| `regression_app.py`            | Streamlit salary prediction application                 |
| `regression_model.keras`       | Trained ANN regression model                            |
| `scaler_reg.pkl`               | Regression feature scaler                               |
| `label_encoder_gender_reg.pkl` | Regression gender encoder                               |
| `onehot_encoder_geo_reg.pkl`   | Regression geography encoder                            |
| `logs/`                        | Classification TensorBoard logs                         |
| `regressionlogs/`              | Regression TensorBoard logs                             |
| `screenshots/`                 | Training and model visualizations                       |
| `requirements.txt`             | Python dependencies                                     |
| `LICENSE`                      | GNU GPL v3 license                                      |

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

Activate the environment:

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

Current dependencies include:

* TensorFlow `2.21.0`
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TensorBoard
* Matplotlib
* IPykernel

---

# ▶️ Run the Applications

## Customer Churn Prediction

Run:

```bash
streamlit run app.py
```

The application allows you to enter customer information and returns:

```text
Churn Probability
        +
Churn Classification
```

---

## Salary Prediction

Run:

```bash
streamlit run regression_app.py
```

The application accepts customer information and returns:

```text
Predicted Estimated Salary
```

---

# 🧪 Training

## Classification

The classification model can be explored and trained using:

```text
experiments.ipynb
```

The workflow includes:

```text
Load Dataset
      ↓
Data Exploration
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Train/Test Split
      ↓
Scaling
      ↓
ANN Construction
      ↓
Training
      ↓
Early Stopping
      ↓
Validation
      ↓
Model Saving
```

The trained classification model is saved as:

```text
model.h5
```

---

# 💰 Regression Training

The salary prediction model is developed in:

```text
salaryregression.ipynb
```

The workflow changes the target variable from:

```text
Exited
```

to:

```text
EstimatedSalary
```

making the problem a regression task.

The trained model is saved as:

```text
regression_model.keras
```

and the corresponding preprocessing artifacts are stored separately.

---

# ⏱️ Early Stopping

The ANN training workflow also includes **early stopping**.

Instead of blindly continuing training for a fixed number of epochs, the model can monitor validation performance and stop training when improvement stops.

Conceptually:

```text
Training
   │
   ├── Validation improves
   │        ↓
   │     Continue
   │
   └── Validation stops improving
            ↓
       Early Stopping
```

This helps reduce unnecessary training and can help limit overfitting.

---

# 📈 TensorBoard

TensorBoard logs are included for both projects.

### Classification

```text
logs/
```

### Regression

```text
regressionlogs/
```

These logs can be inspected using TensorBoard to understand model training and validation behavior.

Launch TensorBoard with:

```bash
tensorboard --logdir logs
```

For the regression experiment:

```bash
tensorboard --logdir regressionlogs
```

---

# 🖥️ Streamlit Deployment

Both trained models have been connected to Streamlit applications.

### Classification

```text
app.py
```

### Regression

```text
regression_app.py
```

This creates a simple bridge between:

```text
Machine Learning Model
        ↓
Saved Model Artifact
        ↓
Inference Pipeline
        ↓
Streamlit Interface
        ↓
User
```

The applications load the trained model and preprocessing artifacts directly rather than retraining the model every time a prediction is requested.

---

# 🔍 Key Learning Objectives

This repository focuses on understanding Artificial Neural Networks from experimentation through deployment.

### Machine Learning

* Binary classification
* Regression
* Artificial Neural Networks
* Model training
* Validation
* Early stopping
* Model persistence
* Inference

### Data Preprocessing

* Feature selection
* Label encoding
* One-hot encoding
* Standardization
* Train/test splitting
* Reusing preprocessing artifacts

### Deep Learning

* TensorFlow
* Keras
* Dense neural network layers
* Training and validation
* TensorBoard monitoring

### Deployment

* Streamlit
* Loading trained models
* Interactive prediction
* Connecting preprocessing pipelines with deployed models

---

# 💡 What This Repository Demonstrates

The main idea behind this repository is not simply to train an ANN.

It demonstrates the journey from:

```text
                    Data
                     ↓
              Experimentation
                     ↓
              Feature Engineering
                     ↓
               ANN Training
                     ↓
                Validation
                     ↓
             Saved Model
                     ↓
              Inference Code
                     ↓
              Streamlit App
```

In other words, the focus is on understanding how a model moves from a **Jupyter notebook into a usable application**.

---

# ⚠️ Important Notes

* Run the Streamlit applications from the **repository root** so that relative model and preprocessing paths resolve correctly.
* The churn application uses a `0.50` probability threshold for classification.
* The regression application predicts a continuous estimated salary value.
* The classification and regression models have separate preprocessing artifacts.
* The repository contains trained model files, so retraining is not required simply to run the applications.
* TensorBoard logs are included for training analysis.
* The models are intended for learning and demonstration purposes and should not be treated as production banking decision systems without additional validation, monitoring, security, fairness analysis, and domain-specific testing.

---

# 🚧 Future Improvements

Some possible next steps for the repository include:

* [ ] Add detailed classification metrics
* [ ] Add ROC-AUC and Precision-Recall curves
* [ ] Add confusion matrix
* [ ] Add regression metrics such as MAE, MSE, RMSE and R²
* [ ] Add model architecture visualizations
* [ ] Add hyperparameter tuning
* [ ] Add experiment tracking
* [ ] Add automated tests
* [ ] Add better input validation
* [ ] Add model explainability
* [ ] Containerize applications with Docker
* [ ] Deploy both applications to the cloud
* [ ] Add CI/CD
* [ ] Add model monitoring
* [ ] Improve Streamlit UI/UX

---

# 🧰 Tech Stack

### Languages

* Python

### Machine Learning

* TensorFlow
* Keras
* Scikit-learn

### Data

* Pandas
* NumPy

### Visualization & Experimentation

* Matplotlib
* Jupyter Notebook
* TensorBoard

### Application

* Streamlit

---

# 📜 License

This project is licensed under the **GNU General Public License v3.0**.

See the [`LICENSE`](./LICENSE) file for the complete license terms.

---

# ⭐ Future Direction

This repository is evolving from a single ANN classification project into a broader collection of neural-network experiments covering different supervised learning problems.

Current projects:

```text
┌──────────────────────────────┐
│     ANN Machine Learning     │
├──────────────────────────────┤
│                              │
│  🏦 Customer Churn           │
│       Classification         │
│                              │
│  💰 Salary Prediction        │
│       Regression             │
│                              │
└──────────────────────────────┘
```

More experiments and applications can be added to the repository as the learning journey continues.

---

<div align="center">

## ⭐ If you find this repository useful, consider giving it a star!

### Built with Python • TensorFlow • Keras • Scikit-learn • Streamlit

</div>
