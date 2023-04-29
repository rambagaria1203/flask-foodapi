from flask import Flask, jsonify, request
from flask import Flask, render_template, request, redirect, send_file, url_for
import torch
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pandas
import cv2
import PIL
from PIL import Image, ImageTk
import supervision
from supervision import Detections, BoxAnnotator
from io import BytesIO
import requests
import io
import tensorflow as tf
#from werkzeug.utils import secure_filename
import os

from ultralytics.yolo.v8.detect.predict import DetectionPredictor

    
app = Flask(__name__)

# Load the pre-trained object detection model
#model = torch.load("D:\food_api\venv\best .pt")
Model=r"D:\food_api\venv\best .pt"
model=YOLO(Model)
# Load the calorie data source
data = pd.read_csv(r"D:\food_api\venv\Food and Calories - Sheet1.csv")

@app.route("/")
def hello_world():
    return render_template('index.html')

# Define the endpoint for the API
@app.route('/api/estimate', methods=['POST'])
def estimate_calories():

    # Get the image data from the request
    image_file = request.files["images"]
    images = Image.open(image_file)
    # Run the object detection model on the image
    result=model(images)[0]
    detection = Detections(
        xyxy=result.boxes.xyxy.cpu().numpy(),
        confidence=result.boxes.conf.cpu().numpy(),
        class_id=result.boxes.cls.cpu().numpy().astype(int)
    )
    # Get the detected object labels
    CLASS_NAMES_DICT = model.model.names
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for xy,_, confidence, class_id, tracker_id
    in detection
    ]
    print(labels)
    # Prompt the user for the volume of each detected food item
    ans = []
    for i in range(len(labels)):
        name, prob = labels[i].split(" ")
        ans.append(name)
    
    
    #foodnum = len(ans)

    #vol_list = input().split(" ")

    # Calculate the total calorie intake
    total_calories = 0

    for food in ans:
        val = (data.loc[data['Food'] == food.capitalize()].loc[:,"Calories"])
        my_str = str(val.iloc[0])
        my_float=float(my_str[:-4])
        my_int=int(my_float)
        total_calories +=int(my_int)
    
    # Return the total calorie intake as a JSON response
    response = {'total_calories': total_calories}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
