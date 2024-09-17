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