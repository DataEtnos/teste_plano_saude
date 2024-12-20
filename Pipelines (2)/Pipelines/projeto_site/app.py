import streamlit as st

st.set_page_config(
    page_title="Insurance Prediction",
    page_icon=".\stethoscope.png"
)

st.sidebar.header('Project Description')

st.write("# Welcome to the Insurance Prediction App 🩺")
st.write("\n\n")

st.image('teste_plano_saude\Pipelines (2)\Pipelines\img\health_insurance_img.jpg')

st.write("\n\n")

st.markdown(
    """
    Medical insurance prediction is crucial in healthcare as it predicts medical costs and helps healthcare organizations prepare for expenses.
    By analyzing factors like demographics, medical history, and lifestyle, insurance companies can set accurate premium rates
    and allocate resources effectively. This also helps high-risk individuals receive necessary resources and support.
    Overall, medical insurance prediction is a valuable tool for both patients and providers in a sustainable healthcare system.

    This App aims to predict the insurance cost using input features like:
    - Age
    - BMI
    - Number of children
    - Smoker status
    """
)

st.success('Please **go to the next pages** to get the predictions.')