#import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#get the data
df = pd.read_csv('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/FYP/DATASET.csv')

#cleaning the data
df = df.replace(['No', 'Yes', 'Not sure', 'Don\'t know', 'Often', 'NA', 'Maybe', 'Rarely', 'Never', 'Sometimes', ], 
                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


#set a subheader
st.subheader('Data Information:')
#show the data as a table
st.dataframe(df)
#show statistics on the data
st.write(df.describe())
#show data as a chart
chart = st.bar_chart(df)

#split the data into independent "x" and dependent "Y" variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the freature input from the user
def get_user_input():
    family_history = st.sidebar.slider('family_history', 0, 17, 3)
    work_interfere = st.sidebar.slider('work_interfere', 0, 17, 3)
    remote_work = st.sidebar.slider('remote_work', 0, 17, 3)
    care_options = st.sidebar.slider('care_options', 0, 17, 3)
    wellness_program = st.sidebar.slider('wellness_program', 0, 17, 3)
    seek_help = st.sidebar.slider('seek_help', 0, 17, 3)
    mental_health_consequence = st.sidebar.slider('mental_health_consequence', 0, 17, 3)
    phys_health_consequence = st.sidebar.slider('phys_health_consequence', 0, 17, 3)

    #store a dictionary
    user_data = {'family_history': family_history,
    'work_interfere':work_interfere,
    'remote_work':remote_work,
    'care_options':care_options,
    'wellness_program':wellness_program,
    'seek_help':seek_help,
    'mental_health_consequence':mental_health_consequence,
    'phys_health_consequence':phys_health_consequence
    }
    #transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user input into a variable
user_input = get_user_input()

#set a subheader and display the users input

st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%' )

#Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader and display the classifcation
st.subheader('Classificaition:')
st.write(prediction)
