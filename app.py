import os
from pathlib import Path
import streamlit as st


def main():
    st.title("Resume Parser and Job Matcher")

    with st.sidebar:
        st.subheader("Settings")
        openai_key_txt = st.text_input("OpenAI Token", placeholder="OpenAI token or env[OPENAI_API_KEY]")


    # Resume parsing section
    st.subheader("Resume Parser")
    pdf_file = st.file_uploader("Upload a PDF file for parsing")
    if pdf_file is not None:
        pass

if __name__ == "__main__":
    main()