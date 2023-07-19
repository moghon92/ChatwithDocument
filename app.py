import streamlit as st
import tempfile
import os
import regex as re
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from openai.error import InvalidRequestError, AuthenticationError

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource(show_spinner=False)
def put_files_in_DB(uploaded_files, openai_api_key):
    if uploaded_files is not None:
        documents = []
        file_names = []
        temp_dir = tempfile.TemporaryDirectory()
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                file_names.append(uploaded_file.name)

                # load into Langchain format
                if temp_file_path.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())
                elif temp_file_path.endswith('.docx') or temp_file_path.endswith('.doc'):
                    loader = Docx2txtLoader(temp_file_path)
                    documents.extend(loader.load())
                elif temp_file_path.endswith('.txt'):
                    loader = TextLoader(temp_file_path)
                    documents.extend(loader.load())

        # todo: del temp dir
        # split docs into text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
            chunk_size=528,
            chunk_overlap=28
        )

        texts = text_splitter.split_documents(documents)

        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, embeddings)

        # Create retriever interface
        retriever = db.as_retriever(search_kwargs={"k": 3})

        st.info("Done processing of : "+", ".join([f for f in file_names]))

        return retriever

    return None

def generate_response(retriever, openai_api_key, query_text):
    if retriever is not None:
        # create empty container for steaming
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)

        # define the model
        llm = ChatOpenAI(temperature=0.0,
                         streaming=True,
                         callbacks=[stream_handler],
                         openai_api_key=openai_api_key)

        # define the conversational Q/A chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            memory=st.session_state.buffer_memory,
            verbose=True
        )

        # retrieve a response from the chain
        response = qa({"question": query_text})

        # clean the reponse to display for user
        res_dict = {"answer": response["answer"], "source_documents": []}
        for i, source in enumerate(response["source_documents"]):
            page = 'N/A'
            if "page" in source.metadata.keys():
                page = source.metadata['page']
            res_dict["source_documents"].append({
                f"snippet-{i}": source.page_content,
                "info": {"source": re.search(r'[^/\\&\?]+\.\w{3,4}', source.metadata['source']).group(0),
                         "page": page}
            })

        # store session state in memory
        st.session_state.requests.append(query_text)
        st.session_state.responses.append(response)

        return res_dict
    return None

def main():
    # intialize session variables
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = \
            ConversationBufferWindowMemory(k=3,
                                           memory_key='chat_history',
                                           output_key='answer',
                                           return_messages=True)


    # Page title
    st.set_page_config(page_title="Chat with Docs", page_icon=":books:")
    st.title('Chat with your Documents :book:')

    # read the user's openAI key
    openai_api_key = st.text_input('OpenAI API Key', placeholder='sk-', type='password')
    # File upload
    uploaded_files = st.file_uploader("Upload your files here"
                                      , type=['pdf', 'docx', 'doc', 'txt']
                                      , accept_multiple_files=True
                                      , disabled=not openai_api_key
                                    )

    try:
        # Load document if file is uploaded
        if len(uploaded_files) > 0:
            with st.spinner('Processing...'):
                retriever = put_files_in_DB(uploaded_files, openai_api_key)

            # Form input and query
            result = []
            with st.form('myform', clear_on_submit=False):
                # Query text
                query_text = st.text_input('type your question:', placeholder='Please provide a short summary.')
                # submit key
                submitted = st.form_submit_button('Submit')
                if submitted and openai_api_key.startswith('sk-'):
                    with st.spinner('Calculating...'):
                        res_dict = generate_response(retriever, openai_api_key, query_text)
                        if res_dict is not None:
                            result.append(res_dict['answer'])
                        del openai_api_key

            if len(result):
                with st.expander('sources'):
                    st.write(res_dict['source_documents'])

    except AuthenticationError:
        st.info('The OpenAI API key you entered is incorrect.\nYou can find your API key at https://platform.openai.com/account/api-keys')
    except InvalidRequestError:
        st.info('The recording is corrupted, please try again')
    except:
        st.info('Error, please refresh an try again')

if __name__ == '__main__':
    main()