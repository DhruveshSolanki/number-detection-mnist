from flask import Flask, render_template, request, url_for
import numpy as np
from PIL import Image
import io
import base64
import sys
import os

sys.path.append(os.path.abspath("./model"))
from model.load import *
from tensorflow import compat

global model
graph = compat.v1.get_default_graph()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_model():
    imgSize = 28, 28
    img = Image.open(request.files['imgUpload'])
    print(img)
    data = io.BytesIO()
    img.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())

    image = img.resize(imgSize, Image.ANTIALIAS)
    image = image.convert('1')
    image_array = np.asarray(image)
    image_array = image_array.reshape(1, 28, 28, 1)
    with graph.as_default():
        model = init()
        out = model.predict(image_array)
    return render_template('result.html', img_data=encoded_img_data.decode('utf-8'), ans=np.argmax(out, axis=1))

if __name__ == '__main__':
    app.run(host=0.0.0.0,port=5000,debug=True,use_reloader=False)
