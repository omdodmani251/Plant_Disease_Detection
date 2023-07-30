import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from diseaseDetection import *
from prePreocess import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    file = request.files['image']

    # Save the file to disk

    output_filename = secure_filename(file.filename)
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    file.save(output_filepath)

    result = predictiv3(output_filepath)

    prediction = result
    os.remove(output_filepath)
    return '<html><body><h1>Results:</h1><p>Prediction: {prediction}</p></body></html>'.format(prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
