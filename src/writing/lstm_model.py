from typing import Dict

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense


class LSTMModel:

    def __init__(
            self,
            sequence_length: int,
            step: int,
            lstm_units: int,
            text_encoder: Dict):
        """

        Args:
            sequence_length:
            step:
            lstm_units:
            text_encoder:
        """
        self.sequence_length = sequence_length
        self.step = step
        self.lstm_units = lstm_units
        self.text_encoder = text_encoder

    def build_model(self) -> Model:

        # input
        number_of_classes = len(self.text_encoder.keys())
        sequences_input = Input(
            shape=(self.sequence_length, number_of_classes),
            dtype="float32",
            name="sequences_input")

        # LSTM layer
        x = LSTM(units=self.lstm_units)(sequences_input)

        # Dense layer to predict the classes
        output = Dense(units=number_of_classes, activation="softmax")(x)

        model = Model(sequences_input, output)
        model.summary()
        return model
