from flask import Flask, request, render_template
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("eye_disease_classifier.h5")

# Define class names
class_names = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Updated to 256x256
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    file_name = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_name = file.filename
            # Ensure the 'static' directory exists
            if not os.path.exists("static"):
                os.makedirs("static")

            img_path = "static/" + file_name
            file.save(img_path)
            img_array = prepare_image(img_path)
            predictions = model.predict(img_array)
            pred_class = class_names[np.argmax(predictions)]
            return render_template(
                "index.html",
                prediction=pred_class,
                img_path=img_path,
                file_name=file_name,
            )
    return render_template("index.html", prediction=None, file_name=file_name)


if __name__ == "__main__":
    app.run(debug=True)
