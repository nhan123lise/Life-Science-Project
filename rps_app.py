import tensorflow as tf
import requests
import base64
model = tf.keras.models.load_model('objects.h5')
model.make_predict_function()

import streamlit as st
st.write("""
         # Data Science project
         """
         )
st.write("This is an image analysis web app for medical and biomedical application")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
dic = {0: "car", 1: "bird", 2: "bucket", 3: "clock"}

def import_and_predict(image_data, model):
        
        image = cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        img_gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img_prediction = np.expand_dims(img_gray,axis=-1)
        img_prediction = np.reshape(img_gray,(28,28,1))
        img_prediction = (255 - img_gray.reshape(1, 28, 28).astype('float32'))/255
        prediction = np.argmax(model.predict(np.array([img_prediction])),axis=-1)
        return jsonify({
            'prediction' : str(dic[prediction[0]]),
            'status' :True
        })
# st.button("Machine Learning")
# st.button("Deep Learning")
if file is None:
    st.text("Please upload an image file")
else:
    if st.button("Machine Learning"):
        with open(file,'wb') as temp:
            temp.write(imgBytes)
        image = Image.open(file)
        image = base64.b64decode(image)
        image = cv2.imread(image)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        st.write(prediction)
    if st.button("Deep Learning"):
        st.write("hihi")
