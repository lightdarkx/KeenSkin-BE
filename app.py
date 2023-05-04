from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')

app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)
# Model saved with Keras model.save()
MODEL_PATH = 'models/model_v1.h5'


#Load your trained model
model = load_model(MODEL_PATH)
        # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')


class_dict = {0:"Actinic Keratosis (Intraepithelial Carcinoma)",
             1:"Basal Cell Carcinoma",
             2:"Benign Keratosis",
             3:"Dermatofibroma",
             4:"Melanoma",
             5:"Melanocytic Nevus",
             6:"Squamous Cell Carcinoma",
             7:"Vascular Lesion",
             8:"None of the others"}

@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['POST'])
def get_image():
    # Get files from the front-end and send results if success
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    print(file.filename)
    
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['TEST'], filename)
        file.save(filepath)
        pred, _ = model_predict(filepath, model)
    return jsonify({'msg': 'media uploaded successfully', 'result': pred})


def model_predict(img_path, model):
    
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    # preds = model.predict(img)[0]
    preds = model.predict(img)[0]
    print(preds)
    prediction = sorted(
      [(class_dict[i], round(j*100, 2)) for i, j in enumerate(preds)],
      reverse=True,
      key=lambda x: x[1]
  )
    return prediction,img

# this section is used by gunicorn to serve the app on Azure and localhost
if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=8080)

