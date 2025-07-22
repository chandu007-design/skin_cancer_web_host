from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = load_model("best_model.keras")

# Mapping of prediction index to class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Load metadata CSV
metadata_df = pd.read_csv("HAM10000_metadata_with_stage_final.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    if file:
        img_path = "uploads/temp.jpg"
        os.makedirs("uploads", exist_ok=True)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]

        stage_info = ""
        if predicted_class == "mel":
            # Predict stage if melanoma
            # Find the closest match in metadata
            image_id = os.path.splitext(file.filename)[0]
            row = metadata_df[metadata_df['image_id'] == image_id]
            if not row.empty and 'stage' in row.columns:
                stage_info = f"Stage: {row.iloc[0]['stage']}"
            else:
                stage_info = "Stage info not found"

        return render_template('result.html', prediction=predicted_class.upper(), stage=stage_info)

if __name__ == '__main__':
    app.run(debug=True)
