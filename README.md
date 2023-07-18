This is a voice-enabled Question and Answer (Q/A) application that allows users to upload documents and utilize a Language Model (LM) for Q/A using Langchain and Streamlit. The app reads the uploaded documents divides them into chunks, embeds each chunk using OpenAI, loads the embeddings into FAISS vector DB. When the user enters a prompt (via voice) it get's converted into text using speech recognition. The model then retrieves the K-NN (K-Nearest Neighbors) text chunks from the vector DB and sends them to the LM for information retrieval. Once data is retrieved, it gets converted back into speech.

The code is hosted on Streamlit and can be viewed in chat.py.

Features


# Text Q/A App

This is a text-based Question and Answer (Q/A) application that allows users to upload documents and utilize a Language Model (LLM) for Q/A using Langchain and Streamlit. The app reads the uploaded documents divides them into chunks, embeds each chunk using OpenAI, loads the embeddings into FAISS vector DB. When the user enters a prompt. The model then retrieves the K-NN (K-Nearest Neighbors) text chunks from the vector DB and sends them to the LM for information retrieval. The code is hosted on Streamlit and can be viewed in `app.py`.

![App Overview](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CJzoMxqFrxrDv2UpZt23ZQ.png)

## Features

- ✅ Voice Enabled.
- ✅ CHAT WITH MULTIPLE DOCUMENTS SIMULTANEOUSLY.
- ✅ CHAT WITH MULTIPLE DOCUMENT TYPES (PDFS, WORD AND TEXT FILES).
- ✅ THE APP NOW KEEPS CONV HISTORY IN MEMORY, SO YOU CAN ASK FOLLOW-UP QUESTIONS.
- ✅ THE APP NOW RETURNS THE SOURCES OF DATA IT USED TO GENERATE THE RESPONSE (INCLUDING DOCUMENT NAME, PAGE NUMBER AND TEXT SNIPPET).
- ✅ YOU CAN NOW WATCH THE OUTPUT AS IT'S BEING GENERATED, MUCH LIKE THE GPT INTERFACE.
- 
## Getting Started

To get started with the Text Q/A App, follow these steps:

1. Obtain OpenAI API key: Visit [OpenAI API](https://platform.openai.com/account/api-keys) and create a new secret key for authentication purposes.

2. Try out the app through the following link: [Text Q/A App](https://chatwithdocument-mohamed.streamlit.app/)

