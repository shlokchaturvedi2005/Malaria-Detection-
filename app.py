from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("malaria_model.h5")

def predict_image(img):
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Parasitized"
    else:
        return "Uninfected"

@app.route("/", methods=["GET","POST"])
def index():

    result=None

    if request.method=="POST":
        file=request.files["image"]
        img=Image.open(file)
        result=predict_image(img)

    return render_template("index.html", result=result)

app.run(debug=True)