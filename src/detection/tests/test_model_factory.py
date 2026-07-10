# Tests for making the model
# 1. Making sure can make valid model
# 2. Testing with invalid model type, make sure it behaves as expected
import pytest

from detection.detection_model.abstract_detection_model import AbstractDetectionModel
from detection.detection_model.model_factory import ModelFactory
from detection.tests.paths import FAKE_MODEL_PATH, TEST_MODEL_PATH


def load_valid_config():
    return {
        "model_type": "yolo",
        "model_path": TEST_MODEL_PATH
    }

def load_fake_config():
    return {
        "model_type": "fake_model",
        "model_path": FAKE_MODEL_PATH
    }

def test_model_factory_valid_config_returns_abstract_detection_model():
    config = load_valid_config()
    model = ModelFactory.get_model(config["model_type"], config["model_path"])
    assert isinstance(model, AbstractDetectionModel)

def test_model_factory_invalid_config_raises_exception():
    config = load_fake_config()
    with pytest.raises(ValueError):
        model = ModelFactory.get_model(config["model_type"], config["model_path"])
        assert False, "Expected an exception to be raised for invalid model type"