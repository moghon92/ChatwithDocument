# Text Q/A App

This is a text-based Question and Answer (Q/A) application that allows users to upload a text file and utilize a Language Model (LM) for Q/A using Langchain and Streamlit. The app reads the text file, divides it into chunks, embeds each chunk using OpenAI, loads the embeddings into Chromadb vector DB. When the user enters a prompt, the model retrieves the K-NN (K-Nearest Neighbors) text chunks from the vector DB and sends them to the LM for information retrieval. The code is hosted on Streamlit and can be viewed in `app.py`.

![App Overview](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CJzoMxqFrxrDv2UpZt23ZQ.png)

## Features

- Upload text file: Users can upload a text file to be processed by the app.
- Q/A with Language Model: Users can enter prompts or questions, and the app will use the LM to provide answers based on the uploaded text file.
- Chunking and Embedding: The app divides the text into chunks and generates embeddings using OpenAI to facilitate efficient information retrieval.
- VeectorDB: The app retrieves the K-NN text chunks from the vector DB to improve answer accuracy and relevancy.
- Streamlit Integration: The app is built using Streamlit, making it interactive and user-friendly.

## Getting Started

To get started with the Text Q/A App, follow these steps:

1. Obtain OpenAI API key: Visit [OpenAI API](https://platform.openai.com/account/api-keys) and create a new secret key for authentication purposes.

2. Try out the app through the following link: [Text Q/A App](https://chatwithtext-mohamed.streamlit.app/)

