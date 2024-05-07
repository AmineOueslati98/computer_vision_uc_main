"""Test of the SsdFineTune class that is derived from ModelFineTune
 class"""
from io import StringIO

from importlib_resources import files
from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch

from core.models.ssd_fine_tune import SsdFineTune
from core.utils import exp_utils
from core.utils.configs import get_config_from_yaml


def test_ssdfinetune(monkeypatch, test_dir):
    """
    verifies that the class type of generated detection model is SSDMetaArch.
    Tests that the detection model has trainable variables.
    Tests that the layers_to_fine_tune is not empty.
    """
    monkeypatch.setattr("sys.stdin", StringIO("SSD MobileNet V2 FPNLite 320x320"))
    experiment_name = "test_experiment_name"
    config_file = exp_utils.create_experiment(
        experiment_name,
        test_dir,
        files("core.tests.fixtures").joinpath("config.yaml"),
        "SSD MobileNet V2 FPNLite 320x320",
    )
    config = get_config_from_yaml(config_file)
    model_config = config.training.model_config
    checkpoint_path = config.training.checkpoint_path
    model_to_fine_tune = SsdFineTune(num_classes=2, input_shape=[320, 320, 3])
    detection_model, layers_to_fine_tune = model_to_fine_tune(
        model_config, checkpoint_path
    )
    assert isinstance(detection_model, SSDMetaArch)
    assert len(detection_model.trainable_variables) > 0
    assert len(layers_to_fine_tune) > 0
