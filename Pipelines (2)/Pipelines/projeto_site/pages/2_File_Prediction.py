import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Prediction", page_icon="teste_plano_saude\Pipelines (2)\Pipelines\img\stethoscope.png")
st.sidebar.header('File Prediction')
st.title("Insurance prediction")

st.markdown("Predict medical insurance based using a csv file:")

# -- Model -- #
with open('teste_plano_saude\Pipelines (2)\Pipelines\data\models\model_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

data = st.file_uploader('Upload your file')
if data:
    df_input = pd.read_csv(data)
    insurance_prediction = model.predict(df_input)
    df_output = df_input.assign(prediction=insurance_prediction)

    st.markdown('Insurance cost prediction:')
    st.write(df_output)
    st.download_button(
        label='Download CSV', data=df_output.to_csv(index=False).encode('utf-8'),
        mime='text/csv', file_name='predicted_insurance.csv'
        )

