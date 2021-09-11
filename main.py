import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import (FastAPI, File, UploadFile)
from tensorflow.python.keras.preprocessing import image as image_preprocessing
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from googletrans import Translator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_model():
    inception_model = InceptionResNetV2(weights='imagenet')
    if inception_model is None:
        print("Model NOT loaded")
    else:
        print("Model  loaded")
    return inception_model


model = load_model()

app = FastAPI()
translator = Translator()


def predict(img: Image.Image):
    global model
    if model is None:
        model = load_model()
    print("imagen ")
    print(img)
    img = np.asarray(img.resize((299, 299)))[..., :3]
    x = image_preprocessing.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    print('Prediction:', decode_predictions(predictions, top=1)[0][0])
    res = decode_predictions(predictions, top=1)[0][0]

    response = []

    resp = {"class": res[1], "confidence": f"{res[2] * 100:0.2f} %"}

    response.append(resp)

    return response


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.get("/")
def home():
    return {"message": "Hello TutLinks.com"}


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    print(file.filename)
    print(extension)
    if not extension:
        return "Image must be jpg or png format!"

    image = read_image_file(await file.read())

    prediction = predict(image)

    prediction_text = prediction[0]['class']
    prediction_text = prediction_text.replace("_", " ")
    print(prediction_text)

    print("X")
    translation = translator.translate(prediction_text, "es")
    translation = translation.text
    print(translation)

    return translation
