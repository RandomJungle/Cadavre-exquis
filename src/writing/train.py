import logging
import os
import re

from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from typing import Dict

from utils.file_utils import load_model_from_json_and_weights, load_json_file
from writing.encoding import slice_word_sequences, encode_sequences, encode_labels, slice_char_sequences
from writing.lstm_model import LSTMModel


def train_lstm_model(params: Dict, full_text: str, model_path=None, weights_path=None):
    """
    Word level model train function. Builds the model, with metrics and checkpoints, import model
    config and train with number of epochs and parameters provided in config

    Args:
        params: json_config for the model
        full_text: full text of data to process
        model_path: optional path to previously saved model
        weights_path: optional path to previously saved weights

    Returns:
        model history
    """

    # load word2int encoder
    str2int_encoder = load_json_file(params['str2int_encoder_path'])

    # Load model from previous training session
    if model_path and weights_path:
        model = load_model_from_json_and_weights(model_path, weights_path)
    # Create new model if no previous one
    else:
        lstm_model = LSTMModel(
            sequence_length=params['sequence_length'],
            step=params['step'],
            lstm_units=params['lstm_units'],
            text_encoder=str2int_encoder
        )
        model = lstm_model.build_model()

    # Set optimizer
    optimizer = RMSprop()

    # Metrics
    precision = Precision()
    recall = Recall()
    categorical_accuracy = CategoricalAccuracy()
    metrics = [precision, recall, categorical_accuracy]

    model.compile(optimizer=optimizer, loss=params['loss'], metrics=metrics, run_eagerly=False)

    # Define callbacks
    if weights_path:
        last_epoch = max([int(re.search(r"weights\.0?(?P<epoch>\d\d?)-", filename).group("epoch"))
                          for filename in os.listdir(params['model_path']) if filename.endswith("hdf5")])
        file_path = params["model_path"] + '/weights.' + str(last_epoch) + '-{epoch:02d}-{val_loss:.2f}.hdf5'
    else:
        file_path = params["model_path"] + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filepath=file_path,
        verbose=1,
        save_freq='epoch')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0)
    callbacks_fit = [checkpoint, reduce_lr]

    # Save model json
    if not model_path:
        with open(params["model_path"] + '/model_result.json', 'w') as json_file:
            json_file.write(model.to_json())

    # encode data according to level
    x, y = extract_data_with_labels(
        full_text, params, str2int_encoder)
    logging.info(f"Generated {len(x)} training data sequences")

    # Fit model
    logging.info('Start training')
    history = model.fit(
        x, y,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=1,
        callbacks=callbacks_fit,
        validation_split=0.2)

    # Print results
    history = history.history
    logging.info(history)
    model.save_weights(params["model_path"] + '/model_result_weights.h5')

    return history['val_categorical_accuracy'], history['val_loss']


def extract_data_with_labels(full_text: str, params: Dict, str2int_encoder: Dict):
    """
    Reads the data source from full text and encode it to the form expected to train the model

    Args:
        full_text: text of the full data
        params: json config of the model
        str2int_encoder: encoder mapping word to integers

    Returns:
        encoded data and labels as tuples
    """
    if params.get("encoding_level") == 'word':
        subtexts, targets = slice_word_sequences(
            full_text, params.get("sequence_length"), params.get("step"))
    else:
        subtexts, targets = slice_char_sequences(
            full_text, params.get("sequence_length"), params.get("step"))
    x = encode_sequences(subtexts, str2int_encoder)
    y = encode_labels(targets, str2int_encoder)
    return x, y

