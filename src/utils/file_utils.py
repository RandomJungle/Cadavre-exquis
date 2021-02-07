import io
import json
import logging
import os

from typing import Dict

from tensorflow import keras
from tensorflow.python.keras import Model


def generate_text(dir_path: str):
    """
    Generate text as string from dir with .txt files
    """
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            with open(os.path.join(dir_path, file), 'r') as text_file:
                yield text_file.read()


def load_model_from_json_and_weights(model_path: str, weights_path: str) -> Model:
    """
    Load model from json model representation and .h5 weight file
    """
    with open(model_path, 'r') as json_file:
        json_string = json_file.read()
    model = keras.models.model_from_json(json_string)
    model.load_weights(weights_path)
    return model


def load_json_file(json_path: str) -> Dict:
    """
    Load char encoder from json file
    """
    if not os.path.isfile(json_path):
        logging.log(level=40, msg=f"path {json_path} does not exists")
    else:
        with io.open(json_path, mode="r", encoding="utf8") as json_file:
            json_dict = json.load(json_file)
        return json_dict
