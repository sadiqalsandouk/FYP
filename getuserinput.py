# Import libraries
import pandas as pd
import streamlit as st


# Get the freature input from the user
def get_user_input():
    st.sidebar.write("___")
    st.sidebar.write("Please answer the following questions:")
    st.sidebar.write("___")
    st.sidebar.write("Key:")
    st.sidebar.write("Yes = 1 |‎‎‎‎‎‎‎‎‎ No = 0")
    self_employed = st.sidebar.slider('Are you self-employed?', 0, 1, 0)
    healthcare_coverage = st.sidebar.slider('Does your employer provide mental health benefits as part of healthcare coverage?', 0, 1, 0)
    seek_help = st.sidebar.slider('Do you know local or online resources to seek help for a mental health disorder?', 0, 1, 0)
    productivity_affected = st.sidebar.slider('Do you believe your productivity is ever affected by a mental health issue?', 0, 1, 0)
    family_history = st.sidebar.slider('Do you have a family history of mental illness?', 0, 1, 0)
    past_mhdisorder = st.sidebar.slider('Have you had a mental health disorder in the past?', 0, 1, 0)
    diagnosed_mhcondition = st.sidebar.slider('Have you been diagnosed with a mental health condition by a medical professional?', 0, 1, 0)
    interferes_work = st.sidebar.slider('If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?', 0, 1, 0)
    age_q = st.sidebar.slider('What is your age?', 0, 99, 0)
    st.sidebar.write("Key (Gender):")
    st.sidebar.write("M = 0 |‎‎‎‎‎‎‎‎‎ F = 1 | Other = 2")
    gender_q = st.sidebar.slider('What is your gender?', 0, 2, 0)
    remote_work = st.sidebar.slider('Do you work remotely?', 0, 1, 0)
    # Store a dictionary
    user_data = {'self_employed': self_employed,
    'healthcare_coverage':healthcare_coverage,
    'seek_help':seek_help,
    'productivity_affected':productivity_affected,
    'family_history':family_history,
    'past_mhdisorder':past_mhdisorder,
    'diagnosed_mhcondition':diagnosed_mhcondition,
    'interferes_work':interferes_work,
    'age_q':age_q,
    'gender_q':gender_q,
    'remote_work':remote_work
    }

    # Transform the data into a data frame

    features = pd.DataFrame(user_data, index=[0])
    return features