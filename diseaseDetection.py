import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

diseases = {2: 'Tomato Late blight',
            9: 'Tomato healthy',
            1: 'Tomato Early blight',
            4: 'Tomato Septoria leaf spot',
            7: 'Tomato Tomato Yellow Leaf Curl Virus',
            0: 'Tomato Bacterial spot',
            6: 'Tomato Target Spot',
            8: 'Tomato Tomato mosaic virus',
            3: 'Tomato Leaf Mold',
            5: 'Tomato Spider mites Two-spotted spider mite'}

def predictDisease(data):
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    val = pickled_model.predict([data])
    return diseases[int(val)]

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('diseasepredictorModel.h5')

# Define the input shape and number of classes
input_shape = [224, 224]

# Define a function to make a prediction on an input image
def predictiv3(image_path):
    img = load_img(image_path, target_size=input_shape[:2])
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = diseases[class_idx]
    return class_name
