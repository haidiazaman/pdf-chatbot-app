from flask import Flask, render_template, request
from markupsafe import escape

app = Flask(__name__,template_folder='templates')

@app.route("/")
def hello_world():
    return render_template("home.html")

# @app.route("/<name>")
# def hello(name):
#     return f"Hello, {escape(name)}!"

# @app.route('/login')
# def login():
#     return 'login'

@app.route('/user_input', methods=['GET','POST'])
def user_input():
    if request.method == 'POST':
        user_query = request.form.get('text')
        return user_query
    
    return render_template('user_input.html')
