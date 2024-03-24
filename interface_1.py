import streamlit as st
import os
from dotenv import load_dotenv
from resume_parser import ResumeParser, pdf_to_string
from resume_template import Resume
import json

# Load environment variables from .env file
load_dotenv()

# Profile Authentication
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

# Function to parse resume
def parse_resume(resume_file, use_openai=False, openai_key=""):
    p = ResumeParser(use_openai, openai_key)
    resume_text = pdf_to_string(resume_file)
    resume = p.extract_resume_fields(resume_text)
    if isinstance(resume, Resume):
        resume = resume.json()
    return resume

def match_resume_job(resume_file, job_text, use_openai=False, openai_key=""):
    p = ResumeParser(use_openai, openai_key)
    resume_text = pdf_to_string(resume_file)
    res = p.match_resume_with_job_description(resume_text, job_text)
    return res

# Streamlit app
def main():
    st.markdown(
        """
        <style>
            .reportview-container {
                background: url('https://png.pngtree.com/thumb_back/fw800/back_our/20190628/ourmid/pngtree-blue-background-with-geometric-forms-image_280879.jpg');
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("SkillSphere 🌐")

    # Initialize session state
    if 'user_credentials' not in st.session_state:
        st.session_state.user_credentials = load_credentials()

    action = st.sidebar.selectbox("Choose Action", ["Login", "Sign-up"])

    if action == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login(st.session_state.user_credentials, username, password):
                st.title(f"Welcome, {username}")
                st.sidebar.success("Login Successful")
                st.session_state.logged_in = True
            else:
                st.sidebar.error("Invalid Credentials")

    elif action == "Sign-up":
        st.session_state.logged_in = False
        st.sidebar.subheader("Sign-up")
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign-up"):
            if signup(st.session_state.user_credentials, new_username, new_password):
                st.sidebar.success("Sign-up Successful")
            else:
                st.sidebar.error("Username already exists")

    # Show content if logged in
    if st.session_state.get('logged_in'):
        action2 = st.sidebar.selectbox("Choose Action", ["Resume Parser", "Recommendation"])

        if action2 == 'Resume Parser':
            with st.sidebar:
                st.subheader("Parser Options")
                resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

            if resume_file is not None:
                st.subheader("Resume Information")
                resume_info = parse_resume(resume_file)
                st.json(resume_info)

                st.subheader("Job Profile")
                job_text = st.text_area("Job Description")
                if st.button("Match Resume"):
                    match_output = match_resume_job(resume_file, job_text)
                    st.text(match_output)

        elif action2 == 'Recommendation':
            st.success("Sign-up Successful")

if __name__ == "__main__":
    main()
