# dataset - chunking - embeddings
# input data in data folder, can be diff pdf files

# load pdfs into dict of key file_path, value list of document objects split by page number

# check

print(len(documents_dict['/Users/I748920/Desktop/llms-learning/pdf-chatbot-app/data/elements-of-statistical-learning-book/chap10.pdf']))
d = documents_dict['/Users/I748920/Desktop/llms-learning/pdf-chatbot-app/data/elements-of-statistical-learning-book/chap10.pdf']
print(len(d[0].page_content))

# chunk the pdfs

sample_query = "for high-dimensional problems, with regards to p and N, in what \
cases can ridge regression exploit the correlation in the features of the dataset?"
sample_gt_context_start_index = 397
sample_gt_context = [c for c in chunks if c.metadata['start_index']==sample_gt_context_start_index][0]

print(sample_query)
print(sample_gt_context.page_content)




# model - setup rag chain

# setup ollama-llm


# simple conversation history store using dict - storage saved to current sess only, no persistence




if name == "main"

    response = conversational_rag_chain.invoke(
        input={
            'input': 'explain ridge regression in context of high dimensional problems'
        },
        config={
            'configurable':{'session_id':'1'}
        }
    )



# archive code

    # # There is no conversation in the beginning, so input_query is directly used for retrieval --> setup the first q-a
    # current_chat_history = get_session_history(session_id)
    # formatted_chat_history = format_chat_history(current_chat_history)

    # Subsequently there are messages in conversation-history -> need to deal with it in loop
    
    # full_user_input = input("User input: ")
    # # Only take the first part of the user input if --show references is used
    # user_input = full_user_input.split("--")[0]

    # # Retrieve documents relevant for the first input_query
    # retrieved_documents = history_aware_retriever.invoke(
    #     {
    #         'chat_history':formatted_chat_history,
    #         'input':user_input
    #     }
    # )

    # response = manual_rag_with_ollama(
    #     retrieved_documents=retrieved_documents, 
    #     formatted_chat_history=formatted_chat_history, 
    #     input_query=user_input, 
    #     ollama_model_name=llm_model_name
    # )

    # # add the response and input query to the current_chat_history
    # current_chat_history.add_user_message(user_input)
    # current_chat_history.add_ai_message(response)

    # # implement the show contexts part for the first query logic
    # print()
    # if len(full_user_input.split("--"))>1: # just check for non empty string will suffice jic of misspelling
    #     print()
    #     print("References:")
    #     for i,d in enumerate(retrieved_documents):
    #         print(f"{i+1} From: page {d.metadata['page']} of {d.metadata['source'].split('/')[-1]}")
    #         print(f"Content: {d.page_content}")
    #         print()
    
