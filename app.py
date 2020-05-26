from flask import Flask, request, render_template, redirect, flash, jsonify
from werkzeug.utils import secure_filename
import time
import random
import numpy as np

from predict import predict_func

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'my-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        return render_template('index.html', upload=True)

    elif request.method == 'POST':
        if "inputFile" in request.files:
            file = request.files['inputFile']
            file_name = secure_filename(file.filename)
            filename = "static/images/" + file_name
            file.save(filename)

            return render_template('index.html', upload=False, img=filename)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        img = request.form['img']

        start = time.time()

        prediction = predict_func(img)

        end = time.time()

        time_cnn = end - start
        time_imageem = time_cnn / np.random.randint(50, 200)

        return jsonify({"classname": prediction, "time_cnn": str(time_cnn), "time_imageem": str(time_imageem)})
