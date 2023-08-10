from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os


# app inital
app = Flask(__name__)
app.static_folder = 'static'

# main page
@app.route('/')
def index():
    return render_template('index.html')


# main
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
