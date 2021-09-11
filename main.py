import os
import tensorflow as tf
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from fastapi import FastAPI
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()


# model = None


@app.on_event("startup")
def load_model():
    inception_model = InceptionResNetV2(weights='imagenet')
    if inception_model is None:
        print("Model NOT loaded")
    else:
        print("Model  loaded")


@app.get("/")
def home():
    load_model()
    return {"message": "Hello working system"}
