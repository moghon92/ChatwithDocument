import streamlit as st
import tempfile
import os
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA


def read_uploaded_files(uploaded_files):
    if uploaded_files is not None:
        temp_dir = tempfile.TemporaryDirectory()
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

        r_text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=32
        )
        # load into Langchain format
        loader = PyPDFDirectoryLoader(temp_dir.name)
        documents = loader.load_and_split(text_splitter=r_text_splitter)
        return documents

    return None

def generate_response(uploaded_files, openai_api_key, query_text):
    # Load document if file is uploaded
    texts = read_uploaded_files(uploaded_files)

    if texts is not None:
        #documents = [uploaded_file.read().decode()]

        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title="Chat with Doc", page_icon=":books:")
st.title('Chat with your Document')

# File upload
#uploaded_file = st.file_uploader('Upload a txt file', type='txt')
uploaded_files = st.file_uploader("Upload your files here", accept_multiple_files=True)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_files)

# Form input and query
result = []
with st.form('myform', clear_on_submit=False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_files and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, openai_api_key, query_text)
            result.append(response)
            # del openai_api_key

if len(result):
    st.info(response)