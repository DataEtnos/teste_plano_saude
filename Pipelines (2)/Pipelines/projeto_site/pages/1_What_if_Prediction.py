import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="teste_plano_saude/Pipelines/img/stethoscope.png"
)

st.sidebar.header('What if Prediction')
st.title("Insurance Prediction")

st.markdown("Predict medical insurance based on the following features:")

# -- Parameters -- #

age = st.number_input(label='Age', value=18, min_value=18, max_value=120)
bmi = st.number_input(label='BMI', value=30.0)
children = st.slider(label='Children', min_value=0, max_value=5)
smoker = st.selectbox(label='Smoker', options=['no', 'yes'])

# -- Model -- #

model_path = 'teste_plano_saude/Pipelines/data/models/model_pipeline.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the path and try again.")
    model = None

def prediction():
    df_input = pd.DataFrame([{ 'age': age, 'bmi': bmi, 'children': children, 'smoker': smoker }])
    return model.predict(df_input)[0]

# Predict
if st.button('Predict'):
    if model:
        try:
            insurance = prediction()
            st.success(f'**Predicted insurance price:** ${insurance:,.2f}')
        except Exception as error:
            st.error(f"Couldn't predict the input data. The following error occurred: \n\n{error}")
    else:
        st.error("Prediction model is not loaded. Please ensure the model file is available.")
