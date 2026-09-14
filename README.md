# Customer Churn Prediction with ANN

A Streamlit application that predicts whether a bank customer is likely to churn using an Artificial Neural Network (ANN) trained on the Churn Modelling dataset.

## Features

- Interactive customer input form built with Streamlit
- Gender label encoding and geography one-hot encoding
- Standardized input features using the training scaler
- Churn probability produced by a trained TensorFlow/Keras model
- Notebook workflow for preprocessing, training, evaluation, and predictions

## Project Structure

```text
.
├── app.py                    # Streamlit prediction app
├── Churn_Modelling.csv       # Customer churn dataset
├── model.h5                  # Trained ANN model
├── scaler.pkl                # Fitted StandardScaler
├── label_encoder_gender.pkl  # Fitted gender encoder
├── onehot_encoder_geo.pkl    # Fitted geography encoder
├── experiments.ipynb         # Data preprocessing and model training
├── predictions.ipynb         # Example prediction workflow
├── test.ipynb                # Additional experiments or tests
├── requirements.txt          # Python dependencies
└── logs/                     # TensorBoard training logs, when generated
```

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows PowerShell, activation can also be run with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the Application

Run the following command from the project root:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser. Enter the customer's details to view the predicted churn probability and classification.

## Model Inputs

The application uses the following customer attributes:

- Geography
- Gender
- Age
- Balance
- Credit score
- Estimated salary
- Tenure
- Number of products
- Credit card status
- Active member status

A probability above `0.5` is classified as likely to churn.

## Training Workflow

To retrain the model or regenerate the preprocessing artifacts:

1. Open `experiments.ipynb`.
2. Run the preprocessing cells to encode categorical features and fit the scaler.
3. Train the ANN with TensorFlow/Keras.
4. Save the generated files as `model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, and `onehot_encoder_geo.pkl`.
5. Start the Streamlit app again with `streamlit run app.py`.

The notebook uses the `Exited` column as the target and drops `RowNumber`, `CustomerId`, and `Surname` before training.

## Notes

- Run the app from the project root so the model and pickle files can be found using their relative paths.
- The model output is a probability between `0` and `1`; it is displayed rounded to two decimal places in the app.
- TensorBoard logs are written under `logs/fit/` when the training notebook is run.

## License

This project does not currently specify a separate license.
