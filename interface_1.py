import streamlit as st
import os
from dotenv import load_dotenv
from resume_parser import ResumeParser, pdf_to_string
from resume_template import Resume
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Function to parse resume
def parse_resume(resume_file, use_openai=False, openai_key=""):
    p = ResumeParser(use_openai, openai_key)
    resume_text = pdf_to_string(resume_file)
    resume = p.extract_resume_fields(resume_text)
    if isinstance(resume, Resume):
        resume = resume.json()
    return resume

# Function to match resume with job description
def match_resume_job(resume_file, job_text, use_openai=False, openai_key=""):
    p = ResumeParser(use_openai, openai_key)
    resume_text = pdf_to_string(resume_file)
    res = p.match_resume_with_job_description(resume_text, job_text)
    return res

# Streamlit app
def main():
    st.title("Resume Parser and Job Matcher")
    
    # Sidebar widgets
    with st.sidebar:
        st.subheader("Parser Options")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        use_openai = st.checkbox("Use OpenAI")
        openai_key = st.text_input("OpenAI Token", value=os.getenv("OPENAI_API_KEY", ""))

    # Main content
    if resume_file is not None:
        st.subheader("Resume Information")
        resume_info = parse_resume(resume_file, use_openai, openai_key)
        st.json(resume_info)

        # Job Matcher
        st.subheader("Job Matcher")
        job_text = st.text_area("Job Description")
        if st.button("Match Resume"):
            match_output = match_resume_job(resume_file, job_text, use_openai, openai_key)
            st.text(match_output)

if __name__ == "__main__":
    main()
