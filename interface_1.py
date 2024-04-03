import streamlit as st
import os
from dotenv import load_dotenv
from resume_parser import ResumeParser, pdf_to_string
from resume_template import Resume
import json
from streamlit_chat import message
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

p = ResumeParser()
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
    resume_text = pdf_to_string(resume_file)
    print("hello")
    resume = p.extract_resume_fields(resume_text)
    print("hello")
    if isinstance(resume, Resume):
        resume = resume.json()
    return resume

# Function to recommend
def recommendation(resume_file, use_openai=False, openai_key=""):
    recommend = p.recommendation_skill_based()
    
    recommend = recommend.json()
    return recommend

def match_resume_job(resume_file, job_text, use_openai=False, openai_key=""):
    p = ResumeParser(use_openai, openai_key)
    resume_text = pdf_to_string(resume_file)
    res = p.match_resume_with_job_description(resume_text, job_text)
    return res

def display_resource_card(resource):
    st.write(f"**Title:** {resource['title']}")
    st.write(f"**Author:** {resource['author']}")
    st.write(f"**Description:** {resource['description']}")
    st.write(f"[Link to resource]({resource['url']})")
    st.markdown("---")  # Add a horizontal line between cards



# Function to Chatbot     
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Lets Start"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
#  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say
    prompt_template = """
    Resume documents are linked as content analyze it answer the question asked .you can also search answer out from context (if needed)
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input_chain(user_question,history):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain.invoke(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    history.append((user_question, response["output_text"]))

    return response["output_text"]





# Job Search


# Streamlit app
def main():
    st.set_page_config('SkillSphere')
   
    st.title("SkillSphere 🌐")
    st.image('logoh-transformed.png')
    st.sidebar.header("MENU")

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
                st.subheader(f"Welcome {username}")
                
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
        st.sidebar.header("Dashboard")
        action2 = st.sidebar.selectbox("Tools", ["Career Navigator","Resume Parser", "Recommend me","Job Search"])

        if action2 == 'Career Navigator':
            initialize_session_state()
            st.markdown("<h1 style=color:DodgerBlue;font-size: 20px;text-align: center>Career Navigator</h1>",unsafe_allow_html=True)

            uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

            if st.button("Submit & Process"):
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)  
                st.success("Done")

            reply_container = st.container()
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Question:", placeholder="Navigate Your Career with SkillSphere", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    with st.spinner('Generating response...'):
                        output = user_input_chain(user_input, st.session_state['history'])

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

                if st.session_state['generated']:
                        with reply_container:
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


        if action2 == 'Resume Parser':
            st.session_state['past'].clear()
            st.session_state['generated'].clear()
            with st.sidebar:
                st.subheader("Parser Options")
                resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

            if resume_file is not None:
                st.markdown("<h1 style=color:DodgerBlue;font-size: 20px;text-align: center>Resume Information</h1>",unsafe_allow_html=True)
                
               
                resume_info = parse_resume(resume_file)
                st.json(resume_info)

                st.subheader("Job Profile")
                job_text = st.text_area("Job Description")
                if st.button("Match Resume"):
                    match_output = match_resume_job(resume_file, job_text)
                    st.markdown(match_output)

        if action2 == 'Recommend me':
            with st.sidebar:
                st.subheader("Submit resume")
                resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

            if resume_file is not None:
                st.markdown("<h1 style=color:DodgerBlue;font-size: 20px;text-align: center>Recommendation for you</h1>",unsafe_allow_html=True)
                recommend_op = recommendation(resume_file)
                data = json.loads(recommend_op)
                data["content"]
                # st.json(recommend_op)


        if action2 == 'Job Search':
            st.markdown("<h1 style=color:DodgerBlue;font-size: 20px;text-align: center>Job Search</h1>",unsafe_allow_html=True)


            df_jobs = p.process_resume()
            # Display recommended jobs as DataFrame
            st.write("Recommended Jobs:")
            st.dataframe(df_jobs[['Job Title','Company Name','Location','Industry','Sector','Average Salary']])


if __name__ == "__main__":
    main()
