##!/usr/bin/env python3
# Firas Akermi 
# 2022/07/12
# python3
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
st.title('Stroke Prediction Web Application')
st.markdown("""
This application uses a machine learning algorithm (Random forest) to predict Heart Strokes.\n
**Credits**
- Application built in `Python` + `Streamlit` by [Firas Akermi](https://www.linkedin.com/in/firas-akermi)
""")
col1, col2 = st.columns(2)
with col1:
    age=st.slider('Age', 0, 100,key='age')
    gender=st.selectbox('Gender', ['Male', 'Female'])
    hypertension=st.selectbox('Hypertension', ['Yes', 'No'])
    heart_dis=st.selectbox('Heart disease', ['Yes', 'No'])
    married=st.selectbox('Ever married', ['Yes', 'No'])
    work=st.selectbox('Work type', ['Private', 'Self-employed', 'Goverment job', 'Work with Children', 'Never worked'])
    resid=st.selectbox('Residence type', ['Urban', 'Rural'])
    smoker=st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])
    bmi=st.number_input('Body Mass Index',step=1.,format="%.2f")
    avg_glu=st.number_input('Average Glucose Level', step=1.,format="%.2f")
def format_input():
    data = {"gender":[],"age":[],'hypertension':[],'heart_disease':[],'ever_married':[],'work_type':[],
    'Residence_type':[],'avg_glucose_level':[],'bmi':[],'smoking_status':[]}
    data['age'].append(age)
    data['gender'].append(gender)
    if hypertension == 'Yes':
        data['hypertension'].append(1)
    elif hypertension == 'No':
        data['hypertension'].append(0)
    if heart_dis == 'Yes':
        data['heart_disease'].append(1)
    elif heart_dis == 'No':
        data['heart_disease'].append(0)
    data['ever_married'].append(married)
    if work=='Goverment job':
        data['work_type'].append('Govt_job')
    elif work=='Work with Children':
        data['work_type'].append('children')
    elif work=='Never worked':
        data['work_type'].append('Never_worked')
    else:
        data['work_type'].append(work)
    data['Residence_type'].append(resid)
    data['smoking_status'].append(smoker)
    data['bmi'].append(bmi)
    data['avg_glucose_level'].append(avg_glu)
    data = pd.DataFrame(data)
    return data
def build_model(input_data):
    # Reads in saved random forest model
    load_model = pickle.load(open('model_random_forest.pkl', 'rb'))
    # Apply model to make predictions
    prob = load_model.predict_proba(input_data)
    prob_output = pd.DataFrame(prob,columns=['No','Yes'])
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.DataFrame(prediction,columns=['Storke prediction'])
    df = pd.concat([input_data,prediction_output], axis=1)
    values=[prob_output["No"][0],prob_output["Yes"][0]]
    names=['No','Yes']
    fig = go.Figure(
    go.Pie(
    labels = names,
    values = values,
    hoverinfo = "label+percent",
    textinfo = "value"
    ))
    st.write(df)
    st.header("Probability of having heart stroke")
    st.plotly_chart(fig)
    st.markdown(filedownload(df), unsafe_allow_html=True)
def filedownload(df):
    csv = df.to_csv(index=False)
    href = f'<a href="data:file/csv;{csv}" download="prediction.csv">Download Predictions</a>'
    return href
with col2:
    if st.button('Predict'):
        with st.spinner("Predecting..."):
            p=format_input()
            build_model(p)