import pandas as pd
import numpy as np
import os
import random
from imutils import paths
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, url_for
import cv2
from werkzeug.utils import secure_filename

app=Flask(__name__)

PEOPLE_FOLDER=os.path.join('static', 'people_photo')

app.config['UPLOAD_FOLDER']=PEOPLE_FOLDER

file_path = '../static/people_photo'


@app.route('/')
def index():
    return render_template('preview.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    
    model=load_model('../static/people_photo/mobilenet.h5')
    if request.method=='POST':
        f=request.files['image']
        
        f.save(os.path.join(file_path, secure_filename(f.filename)))
        
        img=load_img(f, target_size=(224, 224))
        img=img_to_array(img)
        img=np.expand_dims(img, axis=0)
        img=preprocess_input(img)
        
        
        predicted=model.predict(img)
        result=decode_predictions(predicted, top=3)[0]
      
        
        
        
        file_name=os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        
        
    return render_template('preview.html', result=result,  image=file_name)
    




if __name__ == '__main__':
    app.run(debug=True)        
    
