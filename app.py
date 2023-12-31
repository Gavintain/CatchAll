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

@app.route('/modeltestoverview')
def model_performance_test():
    return render_template('modeloverview.html')

@app.route('/modelstatistic')
def model_statistic():
    return render_template('model_statistic.html')

@app.route('/taskdata')
def taskdata():
    return render_template('taskdata.html')

@app.route('/modeltesting')
def modeltesting():
    return render_template('modeltesting.html')


# main
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
