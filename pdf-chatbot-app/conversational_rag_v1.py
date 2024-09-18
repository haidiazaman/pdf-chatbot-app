print()

import os
import glob
import time
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import warnings
# from urllib3.exceptions import NotOpenSSLWarning
# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress NotOpenSSLWarning from urllib3
warnings.filterwarnings("ignore", module='urllib3')


def load_pdfs(file_paths):
    """
    file_paths must end with .pdf
    PyPDFLoader auto splits the pdf into pages, each page is 1 Document object split by page number

    returns a dict of key: file_path and value: list of document objects
    """
    documents_dict = {}   
    for f in tqdm(file_paths):
        loader = PyPDFLoader(file_path = f)
        documents = loader.load()
        documents_dict[f] = documents
    return documents_dict

def chunk_list_of_documents(documents):
    """
    input a list of documents as Document objects

    output a list of chunks as Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100, # using 20% is a good start
        length_function=len,
        is_separator_regex=False,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)    
    return chunks

def get_session_history(session_id: str):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


def create_huggingface_retriever(folder_path,embedding_model_name):
    files_paths = glob.glob(f"{folder_path}/*.pdf")
    print()
    print()

    # load documents from file paths
    print("loading pdfs...")
    documents_dict = load_pdfs(file_paths=files_paths)

    # chunk documents
    print()
    print("chunking documents...")
    all_chunks = []
    for key in tqdm(documents_dict.keys()):
        documents = documents_dict[key]
        chunks = chunk_list_of_documents(documents=documents)
        all_chunks.extend(chunks)
    print(f"number of chunks: {len(all_chunks)}")

    # setup embedding model
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # setup vectordb, using HF embedding model
    start_time=time.time()
    print()
    print("start process of embedding chunks into vector database...")
    vectorstore_hf = InMemoryVectorStore.from_documents(
        documents=all_chunks,
        embedding=hf_embedding_model
    )
    print("all chunks embedded into vector database!",f"time taken: {round(time.time()-start_time,2)}s")

    # setup retrieval and test with a query and gt_context
    retriever_hf = vectorstore_hf.as_retriever(
        search_type='similarity',
        search_kwargs = {'k':5}
    )
    print("retriever created!")

    return retriever_hf


def create_conversational_rag_chain(retriever_hf,llm_model_name):

    print("creating custom RAG Chat LLM...")
    # setup llm chat model using ollama
    llm_model = ChatOllama(
        model=llm_model_name,
        temperature=0 # increase temp for more creative answers
    ) 

    # setup system contextualise input prompt
    system_contextualise_input_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    system_input_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_contextualise_input_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # instantiate the history-aware retriever:
    history_aware_retriever = create_history_aware_retriever(
        llm=llm_model,
        retriever=retriever_hf,
        prompt=system_input_prompt
    )

    # setup system RAG QnA prompt
    system_rag_qna_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    system_rag_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_rag_qna_prompt),
            MessagesPlaceholder("chat_history"),
            ('human',"{input}"),
        ]
    )

    # instantiate qna_chain
    qna_chain = create_stuff_documents_chain(llm=llm_model,prompt=system_rag_prompt)

    # instantiate rag_chain
    rag_chain = create_retrieval_chain(retriever=history_aware_retriever,combine_docs_chain=qna_chain)

    # create overall conversational RAG Chain
    conversational_rag_chain = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    print("RAG Chat LLM creation complete!")
    return conversational_rag_chain


def main(folder_path,embedding_model_name,llm_model_name):
    if embedding_model_name is None:
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    if llm_model_name is None:
        llm_model_name = 'llama3.1'

    # Create the Hugging Face retriever
    retriever_hf = create_huggingface_retriever(folder_path=folder_path,embedding_model_name=embedding_model_name)
    # Create the conversational RAG chain
    conversational_rag_chain = create_conversational_rag_chain(retriever_hf=retriever_hf,llm_model_name=llm_model_name)

    # Initialize the chat loop
    print()
    print()
    print("You can now start chatting with the LLM.")
    print("Add '--show references' to the end of the input to view the retrieved context chunks referenced by the LLM for answer generation.")
    print("Type 'end session' to stop the conversation.")

    session_id = '1'  # Can be replaced with a unique session ID if needed

    while True:
        print()
        print()
        # Get user input
        full_user_input = input("User input: ")

        # Only take the first part of the user input if --show references is used
        user_input = full_user_input.split("--")[0]

        # End session if user types 'end session'
        if user_input.lower() == "end session":
            print("Ending session. Goodbye!")
            break
        
        # Invoke the conversational RAG chain with the user input
        start_time = time.time()
        response = conversational_rag_chain.invoke(
            input={'input': user_input},
            config={'configurable': {'session_id': session_id}}
        )
        
        # Get the model's answer and print it
        answer = response['answer']
        print(f"LLM Assistant: {answer}")
        print(f"time taken: {round(time.time()-start_time,2)}s")

        # Show the references if user requests
        # if full_user_input.split("--")[-1]=="show references":
        if len(full_user_input.split("--"))>1: # just check for non empty string will suffice jic of misspelling
            print()
            print("References:")
            for i,d in enumerate(response["context"]):
                print(f"{i+1} From: page {d.metadata['page']} of {d.metadata['source'].split('/')[-1]}")
                print(f"Content: {d.page_content}")
                print()



if __name__=="__main__":
    print()

    parser = argparse.ArgumentParser(description="Directly use Conversational RAG with custom PDF documents in terminal.")
    # Add arguments
    parser.add_argument("--folder_path", type=str, help="input absolute folder path to folder of pdfs", required=True)
    parser.add_argument("--embedding_model_name", type=str, help="pass the huggingface embedding model name of your choice", required=False)
    parser.add_argument("--llm_model_name", type=str, help="pass the ollama llm model name of your choice", required=False)

    # Parse the arguments
    args = parser.parse_args()

    # create global variable for chat_history_store
    chat_history_store = {}

    # Run the main chat function 
    main(
        folder_path=args.folder_path,
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name
    )