import os
from flask import Flask, redirect, render_template, request, session, url_for
from scripts.rag_app import RAGChatbot 

class RAG_App:
    def __init__(self):
        self.app = Flask(__name__,template_folder='templates')
        self.folder_path = 'temp_data' # hardcode name of folder to save uploaded files to
        self.rag_chatbot = None

        # Define routes after initializing the app object
        # this replaces the @app.route("/") commonly used before the Flask functions
        self.app.add_url_rule('/', view_func=self.home)
        self.app.add_url_rule('/upload_pdfs', view_func=self.upload_pdfs, methods=['GET', 'POST'])
        self.app.add_url_rule('/user_input', view_func=self.user_input, methods=['GET', 'POST'])

    def home(self):
        return render_template("home.html")

    def upload_pdfs(self):
        if request.method == 'POST':
            f = request.files['file']
            filename = os.path.join(self.folder_path,f.filename) # need to ensure the folder temp_date exists
            f.save(filename)
            print(f"pdf file saved to dir: {filename}")

            # setup rag chatbot vector db
            self.rag_chatbot = RAGChatbot(folder_path=self.folder_path)
            self.rag_chatbot.setup_vector_db()

            return redirect(url_for('user_input'))

        return render_template('upload_pdfs.html')

    def user_input(self):
        response = ""

        # run the python logic here
        if request.method == 'POST':
            user_input = request.form.get('text')

            current_chat_history = self.rag_chatbot.get_session_history()
            formatted_chat_history = self.rag_chatbot.format_chat_history(current_chat_history)

            # retrieve documents using history_aware_retriever
            retrieved_documents = self.rag_chatbot.history_aware_retriever.invoke(
                {
                    'chat_history':formatted_chat_history,
                    'input':user_input
                }
            )

            # invoke manual_rag_with_ollama function and show the results, need to store for chat history update
            response = self.rag_chatbot.manual_rag_with_ollama(
                retrieved_documents=retrieved_documents, 
                formatted_chat_history=formatted_chat_history, 
                input_query=user_input, 
                ollama_model_name=self.rag_chatbot.llm_model_name
            )

            # update chat history with latest user input and LLM output - add the input query and response to the current_chat_history
            current_chat_history.add_user_message(user_input)
            current_chat_history.add_ai_message(response)

        return render_template('user_input.html', response=response)
    
    # show entire chat history
    # change the query to be at the bottom

    def run(self):
        self.app.run()

if __name__=="__main__":
    rag_app = RAG_App()
    rag_app.run()

    # archive

    # sample queries
    # do you know about high dimensional problems in statistical learning? yes or no.
    # explain in which cases can ridge regression do it with regards to p and N in high dimensional? --show references