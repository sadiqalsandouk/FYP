# Import classes
import system_hidden
import global_accuracy
import global_classification
# Import libraries
import streamlit as st

def main():
    system_hidden.main()
    
    st.subheader("Prediction: Does the patient require treatment?")
    st.subheader('Key:')
    st.write('0 = No | 1 = Yes')

    float_accuracy_dtree = float(global_accuracy.accuracy_dtree)
    float_accuracy_forest = float(global_accuracy.accuracy_forest)
    float_accuracy_KNN = float(global_accuracy.accuracy_KNN)
    float_accuracy_GNB = float(global_accuracy.accuracy_GNB)
    float_accuracy_MLP = float(global_accuracy.accuracy_MLP)

    if ((float_accuracy_dtree > float_accuracy_forest) and (float_accuracy_dtree > float_accuracy_KNN) and (float_accuracy_dtree > float_accuracy_GNB) and (float_accuracy_dtree > float_accuracy_MLP)):
        st.write(global_classification.RandomForest_prediction)

    elif ((float_accuracy_forest > float_accuracy_dtree) and (float_accuracy_forest > float_accuracy_KNN) and (float_accuracy_forest > float_accuracy_GNB) and (float_accuracy_forest > float_accuracy_MLP)):
        st.write(global_classification.DecisionTree_prediction)

    elif ((float_accuracy_KNN > float_accuracy_dtree) and (float_accuracy_KNN > float_accuracy_forest) and (float_accuracy_KNN > float_accuracy_GNB) and (float_accuracy_KNN > float_accuracy_MLP)):
        st.write(global_classification.KNN_prediction)
    
    elif ((float_accuracy_GNB > float_accuracy_dtree) and (float_accuracy_GNB > float_accuracy_forest) and (float_accuracy_GNB > float_accuracy_KNN) and (float_accuracy_GNB > float_accuracy_MLP)):
        st.write(global_classification.GNB_prediction)
    
    elif ((float_accuracy_MLP > float_accuracy_dtree) and (float_accuracy_MLP > float_accuracy_forest) and (float_accuracy_MLP > float_accuracy_KNN) and (float_accuracy_MLP > float_accuracy_GNB)):
        st.write(global_classification.MLP_prediction)