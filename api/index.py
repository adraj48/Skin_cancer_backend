import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Custom KANLayer Definition ----
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.silu(tf.matmul(inputs, self.kernel) + self.bias)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# ---- Load Your Model ----
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KANLayer': KANLayer})

# ---- Image Preprocessing ----
IMG_SIZE = 224  # Use your model's input size

def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def preprocess_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = remove_hair(img)
    return img.astype(np.float32)

# ---- Flask App Setup ----
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://skin-cancer-frontend.vercel.app"
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img = preprocess_image(file.read())
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    result = "malignant" if pred >= 0.231 else "benign"
    return jsonify({'probability': float(pred), 'result': result})

