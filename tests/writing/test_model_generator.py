import os

from resources import DATA_PATH
from utils.file_utils import load_model_from_json_and_weights, load_json_file
from writing.char_level_model.model_generator import predict_from_model, sample_next_char


def test_model_prediction():
    initial_prompt = "Once upon a time, "
    input_text = initial_prompt
    model = load_model_from_json_and_weights(
        os.path.join(DATA_PATH, "models/04_06-02-21/model_result.json"),
        os.path.join(DATA_PATH, "models/04_06-02-21/model_result_weights.h5")
    )
    json_config = load_json_file(
        os.path.join(DATA_PATH, "confs/char_level_model/04_06-02-21.json")
    )
    char2int_encoder = load_json_file(
        os.path.join(DATA_PATH, "char_encoders/fairytales/char2int.json")
    )
    int2char_encoder = load_json_file(
        os.path.join(DATA_PATH, "char_encoders/fairytales/int2char.json")
    )
    for i in range(300):
        prediction = predict_from_model(
            model, input_text, json_config, char2int_encoder
        )
        next_char = sample_next_char(prediction, int2char_encoder, 0.5)
        input_text += next_char
    assert initial_prompt < input_text
