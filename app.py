from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import psycopg2
import torch
import ultralytics
import os
import json
from PIL import Image


app = Flask(__name__)
app.static_folder = 'static'


def database_initialize(db_info):
    try:
        conn = psycopg2.connect(**db_info)
        cursor = conn.cursor()

        cursor.execute("""Drop TABLE IF EXISTS model_metadata_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS aihub_img_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS aihub_annotation_table CASCADE;""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS model_metadata_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255),
                        size_mb FLOAT,
                        format VARCHAR(255)
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_img_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255),
                        size_mb FLOAT,
                        type VARCHAR(255),
                        split VARCHAR(255),
                        image BYTEA NOT NULL
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_annotation_table (
                        image_id INT REFERENCES aihub_img_table(id),
                        annotation_id SERIAL PRIMARY KEY , 
                        category_id INT,
                        category_str VARCHAR(255),
                        bboxs FLOAT[],
                        segmentation FLOAT[],
                        is_crowd int
                        );""")

        conn.commit()
        conn.close()
        print('database initalize complete')
    except Exception as e:
        print("while db_initialization, got ", e)

def sample_data_initialize():
    image_path = 'data/image/'
    data_path = 'data/sample_data.json'

    classes = ['Motorcycle_Pedestrian Road Violation', 'Motorcycle_No Helmet', 'Motorcycle_Jaywalking', 
    'Motorcycle_Signal Violation', 'Motorcycle_Stop Line Violation', 'Motorcycle_Crosswalk Violation', 
    'Bicycle_Pedestrian Road Violation', 'Bicycle_No Helmet', 'Bicycle_Jaywalking', 
    'Bicycle_Signal Violation', 'Bicycle_Stop Line Violation', 'Bicycle_Crosswalk Violation', 
    'Kickboard_Pedestrian Road Violation', 'Kickboard_No Helmet', 'Kickboard_Jaywalking', 
    'Kickboard_Signal Violation','Kickboard_Crosswalk Violation', 'Kickboard_Passenger Violation']

    try:
        with open(data_path,'r') as samplefile:
            sampledata = json.load(samplefile)
        print()
        conn = psycopg2.connect(**app.db_info)
        cursor = conn.cursor()

        for catid in sampledata:
            for list in sampledata[catid]:
                img_info = list[0]
                img_annotations = list[1]
                id = img_info['id']
                name = img_info['file_name']
                file_type = 'jpg'
                split = 'test'
                image = Image.open(image_path+name)
                image = image.tobytes()

                annotation_id = img_annotations['id']
                category_id = img_annotations['category_id']
                category_str = classes[category_id]
                bboxs = img_annotations['bbox']
                segmentation = []
                is_crowd = 0

                query1 = """
                        INSERT INTO aihub_img_table (id,name,type,split,image) 
                        VALUES (%s,%s,%s,%s,%s);
                        """
                query2 = """
                        INSERT INTO aihub_annotation_table (image_id,annotation_id,category_id,category_str,bboxs,segmentation,is_crowd) 
                        VALUES (%s,%s,%s,%s,%s,%s,%s);
                        """
                cursor.execute(query1,(id,name,file_type,split,psycopg2.Binary(image)))
                cursor.execute(query2,(id,annotation_id,category_id,category_str,bboxs,segmentation,is_crowd))


        conn.commit()

    # 데이터베이스 데이터 저장 확인
    # query = "SELECT * FROM aihub_img_table;"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # for row in result:
    #     print(row)

    # query = "SELECT * FROM aihub_annotation_table;"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # for row in result:
    #     print(row)

        conn.close()
        print('sample_data_loading successful')
    except Exception as e:
        print("while sample_data_loading, got ",e)

def sample_model_initialize():
    folder_path = 'data/model/'
    modellist = ['yolov8.pt']

    classes = ['Motorcycle_Pedestrian Road Violation', 'Motorcycle_No Helmet', 'Motorcycle_Jaywalking', 
    'Motorcycle_Signal Violation', 'Motorcycle_Stop Line Violation', 'Motorcycle_Crosswalk Violation', 
    'Bicycle_Pedestrian Road Violation', 'Bicycle_No Helmet', 'Bicycle_Jaywalking', 
    'Bicycle_Signal Violation', 'Bicycle_Stop Line Violation', 'Bicycle_Crosswalk Violation', 
    'Kickboard_Pedestrian Road Violation', 'Kickboard_No Helmet', 'Kickboard_Jaywalking', 
    'Kickboard_Signal Violation','Kickboard_Crosswalk Violation', 'Kickboard_Passenger Violation']
    

    try:
        for i,filename in enumerate(modellist):
            id = i
            name = filename.split('.')[0]
            size_mb = round((os.path.getsize(folder_path+filename))/(1024*1000),3)
            format = filename.split('.')[1]
        
            conn = psycopg2.connect(**app.db_info)
            cursor = conn.cursor()
            query1 = """
                    INSERT INTO model_metadata_table (id,name,size_mb,format) 
                    VALUES (%s,%s,%s,%s);
                    """
            cursor.execute(query1,(id,name,size_mb,format))
       
        conn.commit()

        # 데이터베이스 데이터 저장 확인
        query = "SELECT * FROM model_metadata_table;"
        cursor.execute(query)
        result = cursor.fetchall()
        for row in result:
            print(row)

        conn.close()
        print('sample_model_loading successful')
    except Exception as e:
        print("while sample_model_loading, got ",e)

def app_initialize(db_info):
    print('app init start')
    # try:
    app.db_info = db_info
    database_initialize(db_info)
    sample_data_initialize()
    sample_model_initialize()
    print('app_initialize successful')
    # except Exception as e:
        # print(e)



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

            ####### model testing section

            ##############################################

            ####### saving model performance to database
            conn = psycopg2.connect(**DB_postgresql)
            cursor = conn.cursor()

            # 모델 데이터를 BLOB로 데이터베이스에 저장
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
    app_initialize(DB_postgresql)
    app.run(debug=True, host=DB_postgresql['host'])

