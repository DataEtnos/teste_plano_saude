import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="teste_plano_saude/Pipelines/img/stethoscope.png"
)

st.sidebar.header('File Prediction')
st.title("Insurance Prediction")

st.markdown("Predict medical insurance costs using a CSV file:")

# -- Model -- #
model_path = r'teste_plano_saude\Pipelines (2)\Pipelines\data\models\model_pipeline.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the path and try again.")
    model = None

# -- File Upload -- #
data = st.file_uploader('Upload your file', type=['csv'])
if data:
    try:
        df_input = pd.read_csv(data)
        if model:
            insurance_prediction = model.predict(df_input)
            df_output = df_input.assign(prediction=insurance_prediction)

            st.markdown('### Insurance Cost Prediction:')
            st.write(df_output)

            st.download_button(
                label='Download CSV',
                data=df_output.to_csv(index=False).encode('utf-8'),
                mime='text/csv',
                file_name='predicted_insurance.csv'
            )
        else:
            st.error("Model is not loaded. Please ensure the model file is available.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

