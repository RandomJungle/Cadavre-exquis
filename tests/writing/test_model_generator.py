import os

from resources import DATA_PATH
from utils.file_utils import load_model_from_json_and_weights, load_json_file
from writing.model_generator import predict_from_model, sample_next


def test_model_prediction():
    initial_prompt = "Once upon a time, "
    input_text = initial_prompt.lower()
    model = load_model_from_json_and_weights(
        os.path.join(DATA_PATH, "models/word_level/01_07-02-21/model_result.json"),
        os.path.join(DATA_PATH, "models/word_level/01_07-02-21/model_result_weights.h5")
    )
    json_config = load_json_file(
        os.path.join(DATA_PATH, "confs/word_level/01_07-02-21.json")
    )
    str2int_encoder = load_json_file(
        json_config.get('str2int_encoder_path')
    )
    int2str_encoder = load_json_file(
        json_config.get('int2str_encoder_path')
    )
    for i in range(300):
        prediction = predict_from_model(
            model, input_text, json_config, str2int_encoder
        )
        next_entry = sample_next(prediction, int2str_encoder, 0.5)
        input_text += f" {next_entry}"
    assert initial_prompt < input_text
