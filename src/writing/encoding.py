import nltk
import numpy as np
import tensorflow as tf

from nltk.tokenize import word_tokenize
from typing import List, Dict

from utils.file_utils import generate_text


def slice_char_sequences(text: str, sequence_length: int, step: int) -> (List[str], List[str]):
    """
    Slice input text into sequences of given sequence_length, separated by
    parameter step, with the next character following the sequence as target.

    Stops when there is not enough characters left in the text to form a full
    sequence_length long sequence. For example in the text 'one-two-', with a
    sequence length of 4 and a step of 3, the two sequences extracted will be
    'one-' with target 't' and '-two' with target '-'.

    Args:
        text: input text to slice
        sequence_length: the length of a sequence
        step: the separation between each sequence

    Returns:
        A tuple of the list of all the sequences extracted, and the list of
        all their corresponding target, working as class labels
    """
    subtexts = [text[i:sequence_length + i] for i in range(0, len(text) - sequence_length, step)]
    targets = [text[i+sequence_length] for i in range(0, len(text) - sequence_length, step)]
    return subtexts, targets


def slice_word_sequences(text: str, sequence_length: int, step: int) -> (List[str], List[str]):
    """
    Slice input text into sequences of sequence_length number of tokens, one
    sequence each step, with the next token following the sequence as target.

    Stops when there is not enough tokens left in the text to form a full
    sequence_length long sequence.

    Args:
        text: input text to slice
        sequence_length: the length in number of tokens of a sequence
        step: the separation in number of tokens between each sequence

    Returns:
        A tuple of the list of all the sequences extracted, and the list of
        all their corresponding target, working as class labels
    """
    tokenized_text = word_tokenize(text.lower())
    sequences = [tokenized_text[i:sequence_length + i]
                 for i in range(0, len(tokenized_text) - sequence_length, step)]
    targets = [tokenized_text[i+sequence_length]
               for i in range(0, len(tokenized_text) - sequence_length, step)]
    return sequences, targets


def create_encoder(chars: List[str]) -> Dict:
    """
    Populate an encoder dict from a list of strings (chars or tokens)
    """
    unique_chars = set(chars)
    return {key: value for (key, value) in enumerate(unique_chars)}


def populate_char_encoder_from_directory(dir_path: str):
    """
    Populate a char encoder dict from all the .txt files in a directory
    """
    unique_chars = []
    for text in generate_text(dir_path):
        unique_chars.extend(list(set(text).difference(unique_chars)))
    char_encoder = create_encoder(unique_chars)
    return char_encoder


def populate_char_encoder_from_full_text(full_text: str) -> Dict:
    """
    Populate a char encoder dict from a string
    """
    return create_encoder(list(set(full_text)))


def populate_word_encoder_from_full_text(full_text: str) -> Dict:
    """
    Populate a word encoder dict from a string
    """
    # TODO : evaluate if need to add abstract tokens for proper nouns, numbers, etc. but needs a mask on text gen
    return create_encoder(word_tokenize(full_text.lower()))


def encode_sequences(sequences: List[str], str2int_encoder: Dict) -> np.ndarray:
    """
    Encode the sequences as an array of one-hot encoded vectors

    Args:
        sequences: sequences of characters from a text
        str2int_encoder: dict encoder with chars mapping to int

    Returns:
        One-hot encoded array of vectors
    """
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequences.append(np.asarray(one_hot_encode_sequence(sequence, str2int_encoder)))
    return np.asarray(encoded_sequences)


def encode_labels(labels: List[str], str2int_encoder: Dict) -> np.ndarray:
    """
    Encode list of single chars as target classes

    Args:
        labels: list of chars
        str2int_encoder: dict encoder

    Returns:
        array of encoded label
    """
    encoded_labels = []
    for label in labels:
        encoded_labels.append(one_hot_encode_char(label, str2int_encoder))
    return np.asarray(encoded_labels)


def one_hot_encode_char(char: str, char2int_encoder: Dict) -> np.array:
    """
    One-hot encode a single char to a vector

    Args:
        char: character to encode
        char2int_encoder: dict encoder with chars mapping to int

    Returns:
        One-hot encoded vector
    """
    char_int = int(char2int_encoder.get(char))
    zeros_array = np.zeros(len(char2int_encoder.keys()), dtype=np.int8)
    zeros_array[char_int] = 1
    return zeros_array


def one_hot_encode_sequence(sequence: str, char2int_encoder: Dict) -> List[np.ndarray]:
    """
    One-hot encode a char sequence

    Args:
        sequence: A sequence of chars
        char2int_encoder: dict encoder with chars mapping to int

    Returns:
        One-hot encoded array of vectors, one vector for each char
    """
    return [one_hot_encode_char(char, char2int_encoder) for char in sequence]


def decode_class_prediction(prediction: int, int2str_encoder: Dict) -> str:
    return int2str_encoder.get(str(prediction))


def pad_sequence(sequence: np.array, json_conf: Dict) -> np.array:
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences=sequence, maxlen=json_conf.get("sequence_length"), padding="pre", dtype='float32')
