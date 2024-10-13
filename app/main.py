from flask import Flask, render_template, request

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Welcome to the custom pdf chatbot application"
    
    # chat function - take in input of user - will be input of the python script
    # output LLM response
    # take in pdfs and pass to python script
    # @app.route('/')
    # def my_form():

    #     return render_template('templates/user_input.html')

    # @app.route('/', methods=['POST'])
    # def my_form_post():
    #     text = request.args.get['text']
    #     processed_text = text.upper()
    #     return processed_text

    # print("from terminal: {processed_}")
    return app



if __name__=='__main__':
    app = create_app()
    app.run()