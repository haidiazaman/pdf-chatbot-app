# pdf-chatbot-app

This project aims to create an app where users can manually load their custom pdfs and chat to it like ChatGPT. This is done via RAG (Retrieval Augmented Generation). Goal is to learn more about LLMs and RAG.

There will be several iterations of the project. Each time adding more functionalities and improvements, slowly moving towards a usable app. 

Note to self: Will need to think about the package dependecy if using Ollama will need to think of potentially using Docker and if create an app, might need to get a EC2 instance to host the app, storage, memory etc. 


To run app: do 
* cd app
* python flask_app.py

# to do
* get familiar with HTML coding
* stream output
* make it work for multiple pdfs upload
* beautfiy the frontend (CSS?)
* show loading... after upload (not important)
* start to evaluate and improve RAG (do in notebook)
* note: need to check the chunk size for the document chunking, currently the embedding model is HF sentence-transformer/all-mpnet-base-v2 which has max_seq_length=384 (i.e. ~280 words) so need to ensure the current chunk size during chunking is <= to that amount)
