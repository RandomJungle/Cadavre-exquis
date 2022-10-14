import os
import pytest

from resources import DATA_PATH
from writing.encoding import slice_char_sequences, create_encoder


def test_labels_are_coming_after_text():
    text = " helloB helloA helloC"
    subtexts, targets = slice_char_sequences(text, 6, 7)
    assert subtexts == [' hello', ' hello', ' hello']
    assert targets == ['B', 'A', 'C']


TEXTS = [
    "aaa",
    "a sequence of a certain amount of chars",
    "another sequence, a little bit longer with more words in it",
]


MAX_LENS = [
    6,
    15,
    22,
    237
]


STEPS = [
    2,
    7,
    19
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("sequence_length", MAX_LENS)
@pytest.mark.parametrize("step", STEPS)
def test_generate_sequences(text, sequence_length, step):
    subtexts, targets = slice_char_sequences(text, sequence_length, step)
    assert all([len(subtext) == sequence_length for subtext in subtexts])
    if len(text) > sequence_length:
        assert subtexts and targets
        assert len(subtexts) == len(targets)


FILES = [
    os.path.join(DATA_PATH, "text_samples/oliver_twist.txt"),
    os.path.join(DATA_PATH, "text_samples/don_quixote.txt")
]


@pytest.mark.parametrize("file", FILES)
def test_create_char_encoder(file):
    with open(file, 'r') as test_file:
        text = list(test_file.read())
    char_encoder = create_encoder(text)
    assert char_encoder
