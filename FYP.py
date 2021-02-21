#import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image
import streamlit as st
import sqlite3 
import hashlib


# Security
#passlib,hashlib,bcrypt,scrypt

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management

conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def main():
    

    menu = ["Home","Login"]#,"Sign Up"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        image = Image.open('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/Resources/1.png')
        image2 = Image.open('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/Resources/2.png')
        st.image(image, use_column_width=True)
        st.image(image2, use_column_width=True)


    elif choice == "Login":
        st.sidebar.subheader("Login Section")
        st.title("Please login using the left sidebar")
        st.subheader("")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:

                st.success("Logged in as {}".format(username))
                

                pagenav = st.selectbox("Page Navigation",["System","Analytics"])
                if pagenav == "Analytics":
                    
                    #get the data
                    df = pd.read_csv('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/FYP/DATASET.csv', keep_default_na=False)
                    #show the data in a table
                    st.subheader('Data:')
                    st.dataframe(df)
                    #cleaning the data
                    df = df.replace(['No', 'Never', 'NA', 'Don\'t know', 'Not sure', 'Rarely', 'Often', 'Sometimes', 'Maybe', 'Yes'], 
                                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

                    #show the new data in a table
                    st.subheader('New Data:')
                    st.dataframe(df)
                    #show statistics on the data
                    st.subheader('Stats:')
                    st.write(df.describe())
                    #show data as a chart
                    st.subheader('Charts:')
                    charts = st.bar_chart(df)
                    

                elif pagenav == "System":
                    
                    #get the data
                    df = pd.read_csv('C:/Users/sadiq/OneDrive/Work/Uni/CS3605 Final Year Project/FYP/DATASET.csv', keep_default_na=False)       
                    #cleaning the data
                    df = df.replace(['No', 'Never', 'NA', 'Don\'t know', 'Not sure', 'Rarely', 'Often', 'Sometimes', 'Maybe', 'Yes'], 
                                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

                    #split the data into independent "x" and dependent "Y" variables
                    X = df.iloc[:, 0:8].values
                    Y = df.iloc[:, -1].values

                    #split the data set into 75% training and 25% testing
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

                    #get the freature input from the user
                    def get_user_input():
                        st.sidebar.write("___")
                        st.sidebar.write("Please answer the following questions:")
                        st.sidebar.write("___")
                        st.sidebar.write("Key:")
                        st.sidebar.write("Yes=1 |‎‎‎‎‎‎‎‎‎ No=0")
                        family_history = st.sidebar.slider('Does the patient have a family history of mental health issues?', 0, 1, 0)
                        work_interfere = st.sidebar.slider('Is the mental health issues interferring with their work?', 0, 1, 0)
                        remote_work = st.sidebar.slider('Is the patient working remotely (WFH)?', 0, 1, 0)
                        care_options = st.sidebar.slider('Does the patient have access to care options?', 0, 1, 0)
                        wellness_program = st.sidebar.slider('Is the patient on any wellness programs?', 0, 1, 0)
                        seek_help = st.sidebar.slider('Is the patient seeking help?', 0, 1, 0)
                        mental_health_consequence = st.sidebar.slider('Are there any mental health concequences?', 0, 1, 0)
                        phys_health_consequence = st.sidebar.slider('Are there any physical health concequences?', 0, 1, 0)

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


                    alg = ['Decision Tree', 'Random Forest']
                    st.subheader('Select Classifer:')
                    pickclassifer = st.selectbox('', alg)
                    if pickclassifer=='Decision Tree':
                        #Create and train the decision tree classifer
                        DecisionTree = DecisionTreeClassifier()
                        DecisionTree.fit(X_train, Y_train)
                        #Show the models metrics
                        st.subheader('Model Test Accuracy Score:')
                        st.write(str(accuracy_score(Y_test, DecisionTree.predict(X_test)) * 100)+'%' )
                        #Store the models predictions in a variable
                        DecisionTree_prediction = DecisionTree.predict(user_input)
                        #Set a subheader and display the classification
                        st.subheader('Classificaition:')
                        st.write(DecisionTree_prediction)

                    if pickclassifer =='Random Forest':
                        #Create and train the random forest model
                        RandomForest = RandomForestClassifier()
                        RandomForest.fit(X_train,Y_train)
                        #Show the models metrics
                        st.subheader('Model Test Accuracy Score:')
                        st.write(str(accuracy_score(Y_test, RandomForest.predict(X_test)) * 100)+'%' )
                        #Store the models predictions in a variable
                        RandomForest_prediction = RandomForest.predict(user_input)
                        #Set a subheader and display the classifcation
                        st.subheader('Classificaition:')
                        st.write(RandomForest_prediction)
                        

            else:
                st.warning("Incorrect Username/Password")

    elif choice == "Sign Up":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Sign Up"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

main()
