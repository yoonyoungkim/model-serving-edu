from flask import Flask, request, jsonify, g, redirect, url_for, flash, render_template, make_response
from flask_cors import CORS, cross_origin
import requests
import os
import datetime

from pathlib import Path
import shutil
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from imageio import imread
import json
import time
import uuid
import logging

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

UPLOAD_FOLDER = os.path.join('static', 'source')
OUTPUT_FOLDER = os.path.join('static', 'result')
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])

covid_pneumo_model = load_model('./models/inceptionv3_saved.h5')

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

prediction=' '
confidence=0
filename='Image_Prediction.png'
image_name = filename

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
	return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')

@app.route('/covid19/api/v1/healthcheck', methods=['GET', 'POST'])
def liveness():
    return 'Covid19 detector API is live!'

def test_rx_image_for_Covid19(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 224.0

    pred = model.predict(img)
    pred_neg = int(round(pred[0][1]*100))
    pred_pos = int(round(pred[0][0]*100))
    
    if np.argmax(pred, axis=1)[0] == 0:
        prediction = 'Covid-19 POSITIVE'
        prob = pred_pos
    elif np.argmax(pred, axis=1)[0] == 2:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Positive'
        prob = pred_pos
    else:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Negative'
        prob = pred_pos

    img_pred_name = prediction+str(prob)+filename+'.png' #prediction+'_Prob_'+str(prob)+'_Name_'+filename+'.png'
    cv2.imwrite('static/result/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob, img_pred_name

@app.route("/query", methods=["POST"])
def query():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        if file and allowed_file(file.filename):
            filename = str(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            image_name = filename

            # detection covid
            try:
                prediction, prob, img_pred_name = test_rx_image_for_Covid19(covid_pneumo_model, img_path, filename)
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], img_pred_name)
                return render_template('index.html', prediction=prediction, confidence=prob, filename=image_name, xray_image=img_path, xray_image_with_heatmap=output_path)
            except:
                return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, xray_image=img_path)
        else:
            return render_template('index.html', name='FILE NOT ALOWED', confidence=0, filename=image_name, xray_image=img_path)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run()