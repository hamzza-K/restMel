import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request 


model = tf.keras.models.load_model('melanoma_model.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)/255
    # img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    return (f"Result is: {float(model.predict(img[None,:,:])[0])*10:.2f}%")


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return predict_result(img)
    

@app.route('/', methods=['GET'])
def index():
    return 'Melanoma Inference by Hamzza K'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')