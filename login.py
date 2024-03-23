import streamlit as st
import json

def save_credentials(credentials):
    with open("credentials.txt", "w") as f:
        json.dump(credentials, f)

def load_credentials():
    try:
        with open("credentials.txt", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def login(credentials, username, password):
    saved_password = credentials.get(username)
    if saved_password == password:
        return True
    else:
        return False

def signup(credentials, username, password):
    if username not in credentials:
        credentials[username] = password
        save_credentials(credentials)
        return True
    else:
        return False

def main():
    st.title("SkillSphere")

    action = st.sidebar.selectbox("Choose Action", ["Login", "Sign-up"])

    credentials = load_credentials()

    if action == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login(credentials, username, password):
                st.success("Login Successful")
                st.sidebar.empty()
            else:
                st.error("Invalid Credentials")

    elif action == "Sign-up":
        st.sidebar.subheader("Sign-up")
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign-up"):
            if signup(credentials, new_username, new_password):
                st.success("Sign-up Successful")
            else:
                st.error("Username already exists")

if __name__ == "__main__":
    main()
