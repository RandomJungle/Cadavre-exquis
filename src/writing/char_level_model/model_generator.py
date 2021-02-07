import numpy as np

from tensorflow.python.keras import Model
from typing import Dict

from writing.char_level_model.encoding import encode_sequences, decode_class_prediction, pad_sequence


def loop_until_sentence_break(
        model: Model,
        input_text: str,
        json_conf: Dict,
        char2int_encoder: Dict,
        int2char_encoder: Dict,
        temperature: float = 1.0):
    """
    Generate model prediction until a sentence break character is encountered.

    Args:
        model: model predicting the next character
        input_text: input text for the model to sequence
        json_conf: dict of json config with model training parameters
        char2int_encoder: dict of char mapping to int used to train the model
        int2char_encoder: dict of int mapping to char used to train the model
        temperature: model temperature, 1.0 is very adventurous but less confident,
            and decreasing to lower makes the model more confident but also more conservative

    Returns:
        Generated sentence once it has reached an end of sentence
    """
    generated_sentence = ""
    next_char = ""
    while next_char not in [".", "!", "?", ";", ":"]:
        predictions = predict_from_model(
            model, input_text + generated_sentence, json_conf, char2int_encoder)
        next_char = sample_next_char(
            predictions, int2char_encoder, temperature)
        generated_sentence += next_char
    return generated_sentence


def predict_from_model(
        model: Model,
        input_text: str,
        json_conf: Dict,
        char2int_encoder: Dict):
    """
    Make one prediction from encoded input with model

    Args:
        model: model predicting the next character
        input_text: input text for the model to sequence
        json_conf: dict of json config with model training parameters
        char2int_encoder: dict of char mapping to int used to train the model

    Returns:
        model softmax prediction as array
    """
    input_encoded = encode_sequences(
        [input_text[-json_conf.get("sequence_length"):]], char2int_encoder)
    input_padded = pad_sequence(input_encoded, json_conf)
    return model.predict(input_padded)


def sample_next_char(predictions: np.array, int2char_encoder: Dict, temperature: float):
    """
    Sample next char from model prediction

    Args:
        predictions: array of softmax class predictions
        int2char_encoder: dict encoder mapping chars to integers
        temperature: model temperature, 1.0 is adventurous but less confident, and decreasing
            to lower makes the model more confident but also more conservative, and prone to
            repeating itself

    Returns:
        selected next character predicted
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probas = np.random.multinomial(1, predictions.flatten(), 1)
    return decode_class_prediction(np.argmax(probas), int2char_encoder)
