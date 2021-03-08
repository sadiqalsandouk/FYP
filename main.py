# Import classes
import analytics
import system
import simple
# Import libraries
from PIL import Image
import streamlit as st
import seaborn as sns
sns.set_theme(style="whitegrid")
import sqlite3
import hashlib

st.set_option('deprecation.showPyplotGlobalUse', False)
# Security
# Passlib,hashlib,bcrypt,scrypt

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
            # If password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))

            if result:
                st.success("Logged in as {}".format(username))

                pagenav = st.selectbox("Page Navigation",["Main", "Advanced","Analytics"])
                
                if pagenav == "Main":
                    simple.main()


                elif pagenav == "Analytics":
                    analytics.main()

                elif pagenav == "Advanced":
                    system.main()

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

if __name__ == '__main__':
    main()