# Import classes
import getuserinput
import global_accuracy
import global_classification
# Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")

def main():
    # Get the data

    df = pd.read_csv('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/FYP/DATASET.csv', keep_default_na=False)
    
    # Cleaning the data (Yes/No)

    df = df.replace(['', 'Unsure', 'Not applicable to me', 'No, I don\'t know any', 'Not eligible for coverage / N/A','No', 'Never', 'I don\'t know', 'I know some', 'Rarely', 'Often', 'Sometimes', 'Maybe', 'Yes', 'Yes, I know several', 'Always'], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    # Cleaning the gender column does not reflect any views, I have attempted to be as aware as possible with the goal to make the models as optimal as possible.
    # Cleaning the data (Gender - Male)
    df = df.replace(['Male', 'Dude', 'Male.', 'cisdude', 'I\'m a man why didn\'t you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ', 'male ', 'MALE' ,'Sex is male' ,'male', 'Male ', 'M', 'm', 'man', 'Male (cis)', 'cis man', 'cisdude' 'MALE', 'cis male', 'Cis Male', 'Cis male', 'Man', 'mail', 'Malr', 'M|'], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Cleaning the data (Gender - Female)
    df = df.replace(['Female', 'female', 'female ', 'Cis female ', ' Female', 'Female (props for making this a freeform field, though)', 'Female ', 'F', 'f', 'fem', 'woman', 'Woman', 'female/woman', 'Cis-woman', 'fm', 'I identify as female.' ], 
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # Cleaning the data (Gender - Other)
    df = df.replace(['non-binary', 'Nonbinary', 'N/A', 'Agender', 'genderqueer woman', 'Unicorn', 'Androgynous', 'Human', 'Fluid', 'Transitioned, M2F', 'AFAB', 'Enby', 'Female or Multi-Gender Femme', 'Other', 'mtf', 'Genderflux demi-girl', 'Other/Transfeminine', 'none of your business', 'nb masculine', 'genderqueer', 'human', 'Queer', 'Genderqueer', 'Bigender', 'Genderfluid', 'Genderfluid (born female)', 'Male (trans, FtM)', 'Transgender woman', 'Cisgender Female', 'Male/genderqueer', 'female-bodied; no feelings about gender', 'male 9:1 female, roughly', 'Female assigned at birth ' ], 
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    feature_cols = ['Are you self-employed?', 'Does your employer provide mental health benefits as part of healthcare coverage?', 'Do you know local or online resources to seek help for a mental health disorder?', 'Do you believe your productivity is ever affected by a mental health issue?', 'Do you have a family history of mental illness?', 'Have you had a mental health disorder in the past?', 'Have you been diagnosed with a mental health condition by a medical professional?', 'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?', 'What is your age?', 'What is your gender?', 'Do you work remotely?' ]
    X = df[feature_cols]
    y = df.loc[df['Have you ever sought treatment for a mental health issue from a mental health professional?']]

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Create dictionaries for final graph
    # Use: methodDict['Stacking'] = accuracy_score
    methodDict = {}
    rmseDict = ()

    # Split the data into independent "x" and dependent "Y" variables
    X = df.iloc[:, 0:11].values
    y = df.iloc[:, -1].values

    # Split the data set into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Store the user input into a variable
    user_input = getuserinput.get_user_input()

    # Set a subheader and display the users input
    st.subheader('User Input:')
    st.write(user_input)
    alg = ['Decision Tree', 'Random Forest', 'K Nearest Neighbors', 'Gaussian Naive Bayes', 'Multi-layer Perceptron Neural Network']
    st.subheader('Select Classifer:')
    pickclassifer = st.selectbox('', alg)
    if pickclassifer=='Decision Tree':

        # Create and train the decision tree classifer
        DecisionTree = DecisionTreeClassifier()
        DecisionTree.fit(X_train, y_train)

        # Show the models metrics
        st.subheader('Model Test Accuracy Score:')
        global_accuracy.accuracy_dtree = (accuracy_score(y_test, DecisionTree.predict(X_test)) * 100)
        st.write(f"{global_accuracy.accuracy_dtree}%")
        
        # Store the models predictions in a variable
        global_classification.DecisionTree_prediction = DecisionTree.predict(user_input)
        
        # Display Confusion Matrix
        st.subheader('Confusion Matrix:')
        DecisionTree_CM = confusion_matrix(y_test, DecisionTree.predict(X_test))
        st.write(DecisionTree_CM)

        # ROC & AUC
        st.subheader('ROC Curve:')
        r_probs = [0 for _ in range(len(y_test))]
        DecisionTree_probs = DecisionTree.predict_proba(X_test)
        DecisionTree_probs = DecisionTree_probs[:, 1]
        r_auc = roc_auc_score(y_test, r_probs)
        DecisionTree_auc = roc_auc_score(y_test, DecisionTree_probs)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        DecisionTree_fpr, DecisionTree_tpr, _ = roc_curve(y_test, DecisionTree_probs)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(DecisionTree_fpr, DecisionTree_tpr, marker='.', label='DecisionTree (AUROC = %0.3f)' % DecisionTree_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        st.write(('AUROC = %.3f' % (DecisionTree_auc)))
        

        # Set a subheader and display the classification
        st.subheader('Classification:')
        st.write(global_classification.DecisionTree_prediction)

    if pickclassifer =='Random Forest':

        # Build forest / feature importances
        RandomForest = ExtraTreesClassifier(n_estimators=250,random_state=0)
        RandomForest.fit(X, y)
        importances = RandomForest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in RandomForest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        labels = []
        for f in range(X.shape[1]):
            labels.append(feature_cols[f])      
            
        # Plotting the feature importances
        plt.figure(figsize=(12,8))
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), labels, rotation='vertical')
        plt.xlim([-1, X.shape[1]])
        st.subheader('Feature Importance:')
        st.pyplot()

        # Show the models metrics
        st.subheader('Model Test Accuracy Score:')
        global_accuracy.accuracy_forest = (accuracy_score(y_test, RandomForest.predict(X_test)) * 100)
        st.write(f"{global_accuracy.accuracy_forest}%")

        # Display Confusion Matrix
        st.subheader('Confusion Matrix:')
        RandomForest_CM = confusion_matrix(y_test, RandomForest.predict(X_test))
        st.write(RandomForest_CM)

        # ROC & AUC
        st.subheader('ROC Curve:')
        r_probs = [0 for _ in range(len(y_test))]
        RandomForest_probs = RandomForest.predict_proba(X_test)
        RandomForest_probs = RandomForest_probs[:, 1]
        r_auc = roc_auc_score(y_test, r_probs)
        RandomForest_auc = roc_auc_score(y_test, RandomForest_probs)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        RandomForest_fpr, RandomForest_tpr, _ = roc_curve(y_test, RandomForest_probs)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(RandomForest_fpr, RandomForest_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % RandomForest_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        st.write(('AUROC = %.3f' % (RandomForest_auc)))

        # Store the models predictions in a variable
        global_classification.RandomForest_prediction = RandomForest.predict(user_input)

        # Display the classifcation
        st.subheader('Classification:')
        st.write(global_classification.RandomForest_prediction)

    if pickclassifer=='K Nearest Neighbors':

        # Create and train the decision tree classifer
        KNN = KNeighborsClassifier()
        KNN.fit(X_train, y_train)

        # Show the models metrics
        st.subheader('Model Test Accuracy Score:')
        global_accuracy.accuracy_KNN = (accuracy_score(y_test, KNN.predict(X_test)) * 100)
        st.write(f"{global_accuracy.accuracy_KNN}%")

        # Display Confusion Matrix
        st.subheader('Confusion Matrix:')
        KNN_CM = confusion_matrix(y_test, KNN.predict(X_test))
        st.write(KNN_CM)

        # ROC & AUC
        st.subheader('ROC Curve:')
        r_probs = [0 for _ in range(len(y_test))]
        KNN_probs = KNN.predict_proba(X_test)
        KNN_probs = KNN_probs[:, 1]
        r_auc = roc_auc_score(y_test, r_probs)
        KNN_auc = roc_auc_score(y_test, KNN_probs)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        KNN_fpr, KNN_tpr, _ = roc_curve(y_test, KNN_probs)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(KNN_fpr, KNN_tpr, marker='.', label='KNN (AUROC = %0.3f)' % KNN_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        st.write(('AUROC = %.3f' % (KNN_auc)))
        

        # Store the models predictions in a variable
        global_classification.KNN_prediction = KNN.predict(user_input)

        # Set a subheader and display the classification
        st.subheader('Classification:')
        st.write(global_classification.KNN_prediction)

    if pickclassifer=='Gaussian Naive Bayes':
        # Create and train the decision tree classifer

        GBN = GaussianNB()
        GBN.fit(X_train, y_train)
        # Show the models metrics
        st.subheader('Model Test Accuracy Score:')
        global_accuracy.accuracy_GBN = (accuracy_score(y_test, GBN.predict(X_test)) * 100)
        st.write(f"{global_accuracy.accuracy_GBN}%")

        # Display Confusion Matrix
        st.subheader('Confusion Matrix:')
        GBN_CM = confusion_matrix(y_test, GBN.predict(X_test))
        st.write(GBN_CM)

        # ROC & AUC
        st.subheader('ROC Curve:')
        r_probs = [0 for _ in range(len(y_test))]
        GBN_probs = GBN.predict_proba(X_test)
        GBN_probs = GBN_probs[:, 1]
        r_auc = roc_auc_score(y_test, r_probs)
        GBN_auc = roc_auc_score(y_test, GBN_probs)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        GBN_fpr, GBN_tpr, _ = roc_curve(y_test, GBN_probs)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(GBN_fpr, GBN_tpr, marker='.', label='GBN (AUROC = %0.3f)' % GBN_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        st.write(('AUROC = %.3f' % (GBN_auc)))

        # Store the models predictions in a variable
        global_classification.GBN_prediction = GBN.predict(user_input)

        # Set a subheader and display the classification
        st.subheader('Classification:')
        st.write(global_classification.GBN_prediction)

    if pickclassifer=='Multi-layer Perceptron Neural Network':

        # Create and train the decision tree classifer
        MLP = MLPClassifier()
        MLP.fit(X_train, y_train)

        # Show the models metrics
        st.subheader('Model Test Accuracy Score:')
        global_accuracy.accuracy_MLP = (accuracy_score(y_test, MLP.predict(X_test)) * 100)
        st.write(f"{global_accuracy.accuracy_MLP}%")

        # ROC & AUC
        st.subheader('ROC Curve:')
        r_probs = [0 for _ in range(len(y_test))]
        MLP_probs = MLP.predict_proba(X_test)
        MLP_probs = MLP_probs[:, 1]
        r_auc = roc_auc_score(y_test, r_probs)
        MLP_auc = roc_auc_score(y_test, MLP_probs)
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        MLP_fpr, MLP_tpr, _ = roc_curve(y_test, MLP_probs)
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(MLP_fpr, MLP_tpr, marker='.', label='MLP (AUROC = %0.3f)' % MLP_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        st.write(('AUROC = %.3f' % (MLP_auc)))
        
        # Display Confusion Matrix
        st.subheader('Confusion Matrix:')
        MLP_CM = confusion_matrix(y_test, MLP.predict(X_test))
        st.write(MLP_CM)

        # Store the models predictions in a variable
        global_classification.MLP_prediction = MLP.predict(user_input)

        # Set a subheader and display the classification
        st.subheader('Classification:')
        st.write(global_classification.MLP_prediction)