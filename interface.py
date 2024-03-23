import os
from pathlib import Path
import streamlit as st

# from resume_parser import ResumeParser, pdf_to_string
# from resume_template import Resume

# Function to parse resume
def parse_resume(resume_file, use_openai=False, openai_key=""):
    pass
    # p = ResumeParser(use_openai, openai_key)
    # resume_text = pdf_to_string(resume_file)
    # resume = p.extract_resume_fields(resume_text)
    # if isinstance(resume, Resume):
    #     resume = resume.json()
    # return resume

# Function to match resume with job description
def match_resume_job(resume_file, job_text, use_openai=False, openai_key=""):
    pass
    # p = ResumeParser(use_openai, openai_key)
    # resume_text = pdf_to_string(resume_file)
    # res = p.match_resume_with_job_description(resume_text, job_text)
    # return res

# Function to change OpenAI checkbox based on API key
def change_openai_checkbox(openai_key:str):
    pass
    # if openai_key.startswith("sk-") or "OPENAI_API_KEY" in os.environ:
    #     return st.checkbox(label="Use OpenAI")
    # else:
    #     return st.checkbox(label="Use OpenAI", value=False, disabled=True)

# Main Streamlit app
def main():
    st.title("Resume Parser and Job Matcher")

    with st.sidebar:
        st.subheader("Settings")
        openai_key_txt = st.text_input("OpenAI Token", placeholder="OpenAI token or env[OPENAI_API_KEY]")
        use_openai_chk = change_openai_checkbox(openai_key_txt)

    # Resume parsing section
    st.subheader("Resume Parser")
    pdf_file = st.file_uploader("Upload a PDF file for parsing")
    if pdf_file is not None:
        parsed_resume = parse_resume(pdf_file, use_openai_chk, openai_key_txt)
        st.json(parsed_resume)

    # Job matching section
    st.subheader("Job Matcher")
    job_text = st.text_area("Enter the job description")
    if job_text:
        match_output = match_resume_job(pdf_file, job_text, use_openai_chk, openai_key_txt)
        st.text("Match Output:")
        st.write(match_output)

if __name__ == "__main__":
    main()