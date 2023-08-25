from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import psycopg2
import torch
import ultralytics
import os


app = Flask(__name__)
app.static_folder = 'static'


def app_initialize(db_info):
    app.db_info = db_info
    db_initialize(db_info)
    print('app_initialize successful')

def db_initialize(db_info):

    conn = psycopg2.connect(**db_info)
    cursor = conn.cursor()

    cursor.execute("""Drop TABLE IF EXISTS model_metadata_table;""")
    cursor.execute("""Drop TABLE IF EXISTS aihub_img_table;""")
    cursor.execute("""Drop TABLE IF EXISTS aihub_annotation_table;""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS model_metadata_table (
                    id SERIAL PRIMARY KEY,
                    name CHAR(100),
                    size_mb FLOAT,
                    format CHAR(100)
                    );""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_img_table (
                    id SERIAL PRIMARY KEY,
                    name CHAR(100),
                    size_mb FLOAT,
                    type CHAR(100),
                    split CHAR(100),
                    image BYTEA NOT NULL
                    );""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_annotation_table (
                    id SERIAL PRIMARY KEY,
                    category_num INT,
                    category_str CHAR(100),
                    bboxs FLOAT[]
                    );""")

    conn.commit()
    conn.close()
    print('database initalize complete')

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

@app.route('/modeltesting', methods=['GET'])
def modeltesting():
    return render_template('modeltesting.html')
    
@app.route('/modelupload', methods=['POST'])
def model_upload():
    # image = request.files['image']
    try:
        uploaded_model = request.files['model']

        model_data = uploaded_model.read()  # 파일 내용을 읽어옴
        model_size = round(len(model_data)/(1024*1024),2)  # 파일 크기를 바이트로 읽고 메가바이트로 변환
        model_filename = uploaded_model.filename
        model_name = model_filename.split('.')[0]
        model_filetype = model_filename.split('.')[1]
        print(model_size)
        if model_filetype == 'pt':
            with open(f"models/temp/{uploaded_model.filename}", "wb") as f:
                f.write(model_data)

            model = torch.load(f"models/temp/{uploaded_model.filename}")

            ####### model testing paze

            ##############################################

            ####### saving model performance to database
            conn = psycopg2.connect(**DB_postgresql)
            cursor = conn.cursor()

            # 이미지와 모델 데이터를 BLOB로 데이터베이스에 저장
            # cursor.execute('INSERT INTO uploads (image, model) VALUES (%s, %s)',
            #                 (psycopg2.Binary(image.read()), psycopg2.Binary(model.read())))

            # cursor.execute('INSERT INTO uploads (image) VALUES (%s)',
            #                 (psycopg2.Binary(image.read()),))


            conn.commit()
            conn.close()
            print('success')
            ##############################################

            del(model)
        else:
            raise('filetype is not supported')
    except Exception as e: 
        print(f"Error: {e}")

    try:
        os.remove(f"models/temp/{uploaded_model.filename}")
    except:
        pass


    ####### showing result on the page

    ##############################################
    return redirect(url_for('modeltesting'))
    

# main
if __name__ == '__main__':
    DB_postgresql = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'ghkanf0810!',
    'host': 'localhost'
    }
    app.run(debug=True, host=DB_postgresql['host'])
    app_initialize()
