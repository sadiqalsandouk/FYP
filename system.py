# import classes
import getuserinput

# import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")

accuracy_forest = ""
accuracy_dtree = ""

def main():
    # get the data

    df = pd.read_csv('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/FYP/DATASET.csv', keep_default_na=False)
    # show the data in a table

    st.subheader('Data:')
    st.dataframe(df)
    # cleaning the data (Yes/No)

    df = df.replace(['', 'Unsure', 'Not applicable to me', 'No, I don\'t know any', 'Not eligible for coverage / N/A','No', 'Never', 'I don\'t know', 'I know some', 'Rarely', 'Often', 'Sometimes', 'Maybe', 'Yes', 'Yes, I know several', 'Always'], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    # cleaning the gender column does not reflect any views, I have attempted to be as aware as possible with the goal to make the models as optimal as possible.
    # cleaning the data (Gender - Male)

    df = df.replace(['Male', 'Dude', 'Male.', 'cisdude', 'I\'m a man why didn\'t you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ', 'male ', 'MALE' ,'Sex is male' ,'male', 'Male ', 'M', 'm', 'man', 'Male (cis)', 'cis man', 'cisdude' 'MALE', 'cis male', 'Cis Male', 'Cis male', 'Man', 'mail', 'Malr', 'M|'], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # cleaning the data (Gender - Female)

    df = df.replace(['Female', 'female', 'female ', 'Cis female ', ' Female', 'Female (props for making this a freeform field, though)', 'Female ', 'F', 'f', 'fem', 'woman', 'Woman', 'female/woman', 'Cis-woman', 'fm', 'I identify as female.' ], 
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # cleaning the data (Gender - Other)

    df = df.replace(['non-binary', 'Nonbinary', 'N/A', 'Agender', 'genderqueer woman', 'Unicorn', 'Androgynous', 'Human', 'Fluid', 'Transitioned, M2F', 'AFAB', 'Enby', 'Female or Multi-Gender Femme', 'Other', 'mtf', 'Genderflux demi-girl', 'Other/Transfeminine', 'none of your business', 'nb masculine', 'genderqueer', 'human', 'Queer', 'Genderqueer', 'Bigender', 'Genderfluid', 'Genderfluid (born female)', 'Male (trans, FtM)', 'Transgender woman', 'Cisgender Female', 'Male/genderqueer', 'female-bodied; no feelings about gender', 'male 9:1 female, roughly', 'Female assigned at birth ' ], 
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    

    feature_cols = ['Are you self-employed?', 'Does your employer provide mental health benefits as part of healthcare coverage?', 'Do you know local or online resources to seek help for a mental health disorder?', 'Do you believe your productivity is ever affected by a mental health issue?', 'Do you have a family history of mental illness?', 'Have you had a mental health disorder in the past?', 'Have you been diagnosed with a mental health condition by a medical professional?', 'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?', 'What is your age?', 'What is your gender?', 'Do you work remotely?' ]
    X = df[feature_cols]
    y = df.loc[df['Have you ever sought treatment for a mental health issue from a mental health professional?']]

    # split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Create dictionaries for final graph
    # Use: methodDict['Stacking'] = accuracy_score
    methodDict = {}
    rmseDict = ()

    # split the data into independent "x" and dependent "Y" variables
    X = df.iloc[:, 0:11].values
    y = df.iloc[:, -1].values

    # split the data set into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # store the user input into a variable
    user_input = getuserinput.get_user_input()

    # set a subheader and display the users input

    st.subheader('User Input:')
    st.write(user_input)

    alg = ['Decision Tree', 'Random Forest']
    st.subheader('Select Classifer:')
    pickclassifer = st.selectbox('', alg)
    if pickclassifer=='Decision Tree':
        #create and train the decision tree classifer

        DecisionTree = DecisionTreeClassifier()
        DecisionTree.fit(X_train, y_train)
        #show the models metrics
        st.subheader('Model Test Accuracy Score:')
        accuracy_dtree = (str(accuracy_score(y_test, DecisionTree.predict(X_test)) * 100)+'%')
        st.write(accuracy_dtree)
        #store the models predictions in a variable

        DecisionTree_prediction = DecisionTree.predict(user_input)
        #set a subheader and display the classification

        st.subheader('Classificaition:')
        st.write(DecisionTree_prediction)

    if pickclassifer =='Random Forest':
        #build forest / feature importances

        RandomForest = ExtraTreesClassifier(n_estimators=250,
                                    random_state=0)

        RandomForest.fit(X, y)
        importances = RandomForest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in RandomForest.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]

        labels = []
        for f in range(X.shape[1]):
            labels.append(feature_cols[f])      
            
        #plotting the feature importances

        plt.figure(figsize=(12,8))
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), labels, rotation='vertical')
        plt.xlim([-1, X.shape[1]])
        st.subheader('Feature Importance:')
        st.pyplot()
        #show the models metrics
        
        st.subheader('Model Test Accuracy Score:')
        accuracy_forest = (str(accuracy_score(y_test, RandomForest.predict(X_test)) * 100)+'%')
        st.write(accuracy_forest)
        #store the models predictions in a variable

        RandomForest_prediction = RandomForest.predict(user_input)
        #display the classifcation

        st.subheader('Classification:')
        st.write(RandomForest_prediction)