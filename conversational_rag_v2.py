print()
# import os
import glob
import time
import argparse
from tqdm import tqdm
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import warnings
# from urllib3.exceptions import NotOpenSSLWarning
# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress NotOpenSSLWarning from urllib3
warnings.filterwarnings("ignore", module='urllib3')
from typing import List
from langchain_core.documents import Document


def load_pdfs(file_paths):
    """
    file_paths must end with .pdf
    PyPDFLoader auto splits the pdf into pages, each page is 1 Document object split by page number
    note that the splitting by page number is not perfect, the actual page number might be +/- 1-2pages.

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
    """
    if session_id exists, function returns the ChatMessageHistory of that session_id.

    if session_id does not exists, function instantiates a new ChatMessageHistory.
    """
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

def create_huggingface_retriever(folder_path,embedding_model_name):
    """
    folder_path is type str, absolute folder path of the pdf files' location.
    embedding_model_name type str, take key from HG website.

    1. uses load_pdfs and chunk_list_of_documents functions to
    get chunks across the different input pdfs.
    2. sets up huggingface embedding model based on embedding_model_name passed
    3. sets up the vector db and adds the embeds the chunks into the vector db.
    4. sets up retriever object from the filled vector db

    output: retriever_hf created from the HugginFaceEmbeddings
    """
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
    print()
    print("instantiating HuggingFaceEmbeddings...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("hf_embedding_model created!")

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

def get_retrieved_documents_with_scores(vectorstore,query,k) -> List[Document]:
    """
    vectorstore: vectorstore object
    query (str): user input query
    k (int): number of top K documents to return
    
    manual function to get retrieved documents from vectorstore and add score to metadata.
    output: List of Document objects.
    """
    docs, scores = zip(*vectorstore.similarity_search_with_score(query=query,k=k))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

def instantiate_history_aware_retriever(retriever_hf,llm_model_name):
    """
    retriever_hf is the output of the create_huggingface_retriever function.
    llm_model_name is type str, can be taken from langchain website.

    use langchain to create a history_aware_retriever. 
    The LLM used here is from langchain, not Ollama.

    output: langchain history_aware_retriever object.
    """

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

    return history_aware_retriever

def format_chat_history(chat_history):
    """
    input: chat_history is a langchain_community ChatMessageHistory object.

    function formats the messages into dict format 
    instead of Langchain AI and HumanMessage objects.
    this function returns empty list if there is no chat_history
    """

    formatted_chat_history = [
        {"role": "human", "content": message.content} if isinstance(message, HumanMessage) else
        {"role": "ai", "content": message.content}
        for message in chat_history.messages
    ]

    return formatted_chat_history

def manual_rag_with_ollama(retrieved_documents, formatted_chat_history, input_query, ollama_model_name="llama3.1"):
    """
    Manually performs RAG using retrieved documents from history-aware-retriever and streams results from the Ollama model.
    
    Args:
        retrieved_documents (list of Document objects):  output of history-aware-retriever.invoke().
        formatted_chat_history (list of dict): output of format_chat_history function.
        input_query (str): The user's input query.
        ollama_model_name (str): The name of the Ollama model to use.
    """
    
    # Step 1: Format the retrieved documents as context
    retrieved_references = "\n\n".join([doc.page_content for doc in retrieved_documents])
    
    # Step 2: Create a prompt that integrates the retrieved context and input query
    input_prompt = (
        f"You are an assistant for question-answering tasks. You must reference information from the retrieved_references to answer the input_query. "
        f"You must also reference the formatted_chat_history to take into account conversation flow and to ensure that the response is relevant to both the current query and prior conversation. "
        f"Use five sentences maximum and keep the answer concise. Also, if the input_query is specifically a yes or no question, you must only answer yes or no."
        "\n\n"
        f"retrieved_references: \n{retrieved_references}"
        "\n\n"
        f"formatted_chat_history: \n{formatted_chat_history}"
        "\n\n"
        f"input_query: \n{input_query}"
    )

    # Step 3: Pass the prompt to the Ollama LLM and stream the response
    # print("Streaming response from Ollama...")
    print("LLM Response:")

    stream = ollama.chat(
        model=ollama_model_name,
        messages=[{'role': 'user', 'content': input_prompt}],
        stream=True
    )
    response = ''
    # Stream and display the output from Ollama as it generates
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        response += chunk['message']['content']  # Append each chunk to the answer

    return response

def main(folder_path,embedding_model_name,llm_model_name):
    """
    folder_path (str): absolute folder path of the pdf files' location.
    embedding_model_name (str): model key name that can be taken from HG website.
    llm_model_name (str): model key name that can be taken from Ollama website.

    main function that implements the entire RAG process. 
    session_id is hard-coded to '1' for now since there is no persistence.
    """

    session_id = "1" # hardcode this for temp fix since there is no persistence implemented

    # Initialise embedding and llm model_name
    if embedding_model_name is None:
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    if llm_model_name is None:
        llm_model_name = 'llama3.1'

    # Create the Hugging Face retriever
    retriever_hf = create_huggingface_retriever(folder_path=folder_path,embedding_model_name=embedding_model_name)
    # Create the history-aware-retriever
    history_aware_retriever = instantiate_history_aware_retriever(retriever_hf=retriever_hf,llm_model_name=llm_model_name)

    print()
    print("########################################")
    print("########## START CONVERSATION ##########")
    print("########################################")
    print()
    print("You can now start chatting with the LLM.")
    print("Add '--show references' to the end of the input to view the documents referenced by the LLM.")
    print("Type 'exit' to stop the conversation.")
    print()

    # Initialize the chat loop
    while True:
        # Get user input
        full_user_input = input("User input: ")
        # Only take the first part of the user input if --show references is used
        user_input = full_user_input.split("--")[0]

        # End session if user types 'exit'
        if user_input.lower() == "exit":
            print("Ending session. Goodbye!")
            break

        # get current chat history
        current_chat_history = get_session_history(session_id)
        # format current chat history
        formatted_chat_history = format_chat_history(current_chat_history)

        # retrieve documents using history_aware_retriever
        retrieved_documents = history_aware_retriever.invoke(
            {
                'chat_history':formatted_chat_history,
                'input':user_input
            }
        )

        # invoke manual_rag_with_ollama function  and show the results, need to store for chat history update
        response = manual_rag_with_ollama(
            retrieved_documents=retrieved_documents, 
            formatted_chat_history=formatted_chat_history, 
            input_query=user_input, 
            ollama_model_name=llm_model_name
        )

        # update chat history with latest user input and LLM output - add the input query and response to the current_chat_history
        current_chat_history.add_user_message(user_input)
        current_chat_history.add_ai_message(response)
        
        # Show the references if user requests
        print("\n")
        if len(full_user_input.split("--"))>1: # just check for non empty string will suffice jic of misspelling
            print("### preparing references... ###")
            time.sleep(1)
            print("References:")
            for i,d in enumerate(retrieved_documents):
                time.sleep(1)
                print(f"{i+1} From: page {d.metadata['page']} of {d.metadata['source'].split('/')[-1]}")
                print(f"Content: {d.page_content}")
                print()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Directly use Conversational RAG with custom PDF documents in terminal.")
    # Add arguments
    parser.add_argument("--folder_path", type=str, help="input absolute folder path to folder of pdfs", required=True)
    parser.add_argument("--embedding_model_name", type=str, help="pass the huggingface embedding model name of your choice", required=False)
    parser.add_argument("--llm_model_name", type=str, help="pass the ollama llm model name of your choice", required=False)
    # note that the llm_model_name here is used as the model key for both the langchain ChatOllama model for the history_aware_retriever 
    # and python Ollama model for the LLM answer generation, if a diff key is used and both packages have different names, there will be issues

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

# Issues
# current bottleneck is the history_aware_retriever generating contexts with conversation history reference, but 2-3s shud be fine just like chatgpt
# 3. if cannot find reelvant references, dont return any 
    # - context refernces must be > sim_score, cannot just take top K
# check for hallucinations then return hard coded output like IBM code

# sample queries
# do you know about high dimensional problems in statistical learning? yes or no.
# explain in which cases can ridge regression do it with regards to p and N in high dimensional? --show references

# sample terminal call
# python3 conversational_rag_v2.py --folder_path /Users/I748920/Desktop/pdf-chatbot-app/data/short-chap18-elements-of-statistical-learning-book