import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
import signal

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title('📊 Customer Churn Prediction Dashboard')

# 1. Load the trained model and preprocessing assets safely
@st.cache_resource
def load_assets():
    # Load your updated best model found via Grid Search
    model = tf.keras.models.load_model('optimal_model.h5')
    
    with open('label_encoder_gender_hpt.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encode_geo_hpt.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler_hpt.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
    st.success("Model and preprocessing assets loaded successfully!")
except Exception as e:
    st.error(f"Error loading application assets: {e}")
    st.stop()

# 2. Sidebar: Display Best Performance Metrics from your Grid Search
st.sidebar.header("📊 Model Specifications")
st.sidebar.markdown("""
**Optimal Configuration Found:**
- **Accuracy:** ~85.65%
- **Hidden Layers:** 1
- **Neurons:** 16
- **Training Epochs:** 100
""")
st.sidebar.caption("💡 Run `tensorboard --logdir logs/fit/` in your notebook to view charts.")

st.sidebar.write("---")
# System Shutdown Button
if st.sidebar.button("🛑 Exit Application", use_container_width=True):
    st.sidebar.error("Shutting down web server...")
    st.toast("Application closed! You can close this browser tab.")
    # Sends a termination signal to close the local streamlit server process safely
    os.kill(os.getpid(), signal.SIGINT)

# 3. User Input Layout
st.subheader("🔮 Input Customer Information")
st.write("Modify the parameters below to compute churn risk:")

# Organize into a clean two-column layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 95, value=35)
    balance = st.number_input('Account Balance ($)', min_value=0.0, value=0.0, step=500.0)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)

with col2:
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0)
    tenure = st.slider('Tenure (Years)', 0, 10, value=5)
    num_of_products = st.slider('Number of Products Registered', 1, 4, value=1)
    has_cr_card = st.selectbox('Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member?', ['No', 'Yes'])

# Convert 'Yes'/'No' dropdowns to binary integers (0 or 1) expected by the scaler
has_cr_card_int = 1 if has_cr_card == 'Yes' else 0
is_active_member_int = 1 if is_active_member == 'Yes' else 0

# 4. Process Input Data (Must mirror training dataset column layout exactly)
input_data = pd.DataFrame({
    'CreditScore': [credit_score], 
    'Gender': [label_encoder_gender.transform([gender])[0]], 
    'Age': [age], 
    'Tenure': [tenure], 
    'Balance': [balance], 
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_int], 
    'IsActiveMember': [is_active_member_int], 
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine numerical and categorical features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the final formatted row
input_data_scaled = scaler.transform(input_data)

# 5. Prediction Action Trigger
st.write("---")
if st.button("Calculate Churn Risk", type="primary"):
    # Run forward pass through the newly optimized model
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    # Display Metrics
    st.subheader("🎯 Analysis Results")
    st.metric(label="Calculated Churn Probability", value=f"{prediction_proba * 100:.2f}%")

    if prediction_proba > 0.5:
        st.success("🎉 **Safe Status:** The customer is highly likely to churn.")
    else:
        st.error("🚨 **High Risk Warning:** The customer may not churn.")
