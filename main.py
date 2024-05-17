import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

# Load the trained model
model = load_model(r'cherry_health_classifier.h5')

# Define the function to classify an image
def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Scale pixel values to [0, 1]

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Map the predicted class to a label
    if predicted_class == 0:
        label = 'Healthy'
    else:
        label = 'Diseased'

    return label

# Define the Streamlit app
st.set_page_config(page_title='Cherry Health Classifier', page_icon=':cherry:', layout='wide')
st.title('Cherry Health Classifier')
st.write('Upload an image to classify its health status.')

# Add a file uploader
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

# Add a classify button
if uploaded_file is not None:
    if st.button('Classify'):
        label = classify_image(uploaded_file)
        st.write(f'The image is {label}.')