import json
import logging
import re
import time
from typing import Optional

import os
import typer
from nltk.corpus import gutenberg

from utils.file_utils import load_json_file, load_model_from_json_and_weights
from writing.char_level_model.encoding import populate_char_encoder_from_full_text
from writing.char_level_model.model_generator import loop_until_sentence_break
from writing.char_level_model.train import train_lstm_model

app = typer.Typer()
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


@app.command()
def encode_char(
        output_dir_path: str = typer.Argument(..., help="path to write char encoder json into")):
    with open("/home/juliette/Projects/Cadavre exquis/data/text_samples/fairy_tales.txt", 'r') as input_text_file:
        full_text = input_text_file.read()
    int2char_encoder = populate_char_encoder_from_full_text(full_text)
    char2int_encoder = dict()
    for key, value in int2char_encoder.items():
        char2int_encoder.update({value: key})
    with open(os.path.join(output_dir_path, "int2char.json"), 'w+') as int2char_file, \
            open(os.path.join(output_dir_path, "char2int.json"), 'w+') as char2int_file:
        int2char_file.write(json.dumps(int2char_encoder))
        char2int_file.write(json.dumps(char2int_encoder))


@app.command()
def train(
        config_path: str = typer.Argument(..., help="path to configuration file"),
        model_path: Optional[str] = typer.Argument(None, help="path to previously trained model"),
        weights_path: Optional[str] = typer.Argument(None, help="path to previous weights"),
        use_gutenberg: Optional[bool] = typer.Argument(False, help="whether to use gutenberg corpus for training")):
    config = load_json_file(config_path)
    if use_gutenberg:
        full_text = ' '.join(
            [gutenberg.raw(file_id) for file_id in config.get("train_set")])
    else:
        with open(config.get("text_path"), 'r') as input_text_file:
            full_text = input_text_file.read()
    train_lstm_model(config, full_text, model_path, weights_path)


@app.command()
def play(
        config_path: str = typer.Argument(..., help="path to configuration file"),
        temperature: Optional[float] = typer.Argument(1.0, help="temperature for text generation"),
        model_path: Optional[str] = typer.Argument(None, help="path to previously trained model"),
        weights_path: Optional[str] = typer.Argument(None, help="path to previous weights")):
    config = load_json_file(config_path)
    if not model_path:
        model_path = os.path.join(config.get("model_path"), "model_result.json")
    if not weights_path:
        weights_path = os.path.join(config.get("model_path"), "model_result_weights.h5")
    model = load_model_from_json_and_weights(model_path, weights_path)
    char2int_encoder = load_json_file(config.get("char2int_encoder_path"))
    int2char_encoder = load_json_file(config.get("int2char_encoder_path"))
    prompt = input("Let's play a game. You start by writing a first sentence :\n\n")
    generated_text = ""
    while not re.match("the end", prompt, flags=re.I) and not re.match("the end", generated_text, flags=re.I):
        generated_text = loop_until_sentence_break(
            model, prompt, config, char2int_encoder, int2char_encoder, temperature)
        time.sleep(5)
        prompt = input(generated_text + "\n")


if __name__ == "__main__":
    app()
