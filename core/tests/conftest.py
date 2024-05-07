"""Fixtures used in different tests"""

# pylint: disable=redefined-outer-name

import json
import os
from dataclasses import asdict

import cv2
import numpy as np
import pytest
import yaml
from importlib_resources import files

from core.utils import exp_utils
from core.utils.configs import get_config_from_yaml
from core.utils.inference_utils import restore_weights_from_checkpoint
from core.workflows.train import train


@pytest.fixture(scope="session")
def test_dir(tmpdir_factory):
    """
    Creating a temporary directory where we will put our experiments
    """
    return tmpdir_factory.mktemp("experiment_management")


@pytest.fixture(scope="session")
def experiment_config_file(test_env, test_dir, experiment_name):
    """
    This function defines the config file for the experiment.
    """
    with open(test_dir.join("config.yaml"), "w") as cfg:
        config_dict = asdict(test_env)
        config_dict["ocr"]["preprocessing"]["threshold_mode"] = config_dict["ocr"][
            "preprocessing"
        ]["threshold_mode"].value
        yaml.dump(config_dict, cfg)
    experiment_config_file_path = exp_utils.create_experiment(
        experiment_name,
        test_dir,
        test_dir.join("config.yaml"),
        "SSD MobileNet V2 FPNLite 320x320",
    )
    train(experiment_config_file_path)
    config_file = get_config_from_yaml(experiment_config_file_path)

    return config_file


@pytest.fixture(scope="session")
def test_env(test_dir, input_image, input_bboxes, input_category_ids):
    """This function prepare the folders and files needed for the evaluation."""

    test_dir.mkdir("test_images")
    test_dir.mkdir("test_annotations_json")
    test_dir.mkdir("train_images")
    test_dir.mkdir("train_annotations_json")

    train_image_dir = os.path.join(test_dir, "train_images")
    test_image_dir = os.path.join(test_dir, "test_images")

    train_annotation_path = os.path.join(
        test_dir, "train_annotations_json", "train.json"
    )

    test_annotation_path = os.path.join(test_dir, "test_annotations_json", "test.json")

    test_annotations_dict = {"images": []}
    test_images_list = test_annotations_dict["images"]
    image_names = ["01.jpg", "02.jpg"]
    for image_name in image_names:
        cv2.imwrite(os.path.join(train_image_dir, image_name), input_image)
        cv2.imwrite(os.path.join(test_image_dir, image_name), input_image)

        test_annotations_dict["images"] = test_images_list.append(
            {
                "file_name": image_name,
                "height": input_image.shape[0],
                "width": input_image.shape[1],
                "id": 1,
                "bbox": input_bboxes,
                "category_id": input_category_ids,
                "text_groundtruth": "",
            }
        )

        test_annotations_dict["images"] = test_images_list
        with open(train_annotation_path, "w") as outfile:
            json.dump(test_annotations_dict, outfile)

        with open(test_annotation_path, "w") as outfile:
            json.dump(test_annotations_dict, outfile)

    config = get_config_from_yaml(files("core.tests.fixtures").joinpath("config.yaml"))
    config.data.images_train_dir = train_image_dir
    config.data.annotations_train_file = train_annotation_path

    return config


@pytest.fixture(scope="session")
def experiment_name():
    """Name of the  created experiment"""
    return "experiment_test"


@pytest.fixture(scope="session")
def input_image():
    """ This function defines the input image that we will use during the tests."""
    image_array_1 = np.ones((6, 12, 3), dtype=np.uint8)
    image_array_2 = np.ones((6, 12, 3), dtype=np.uint8) * 2
    image_array = np.vstack((image_array_1, image_array_2))

    return image_array


@pytest.fixture(scope="session")
def input_bboxes():
    """
    This function defines the input bounding boxes that we will use during the tests.
    """
    bboxes = [[0, 0, 4, 5], [9, 5, 11, 7]]

    return bboxes


@pytest.fixture(scope="session")
def input_category_ids():
    """
    This function defines the input category_ids that we will use during the tests.
    """
    category_ids = [0, 0]

    return category_ids


@pytest.fixture(scope="session")
def input_file_name():
    """
    This function defines the input file_name that we will use during the tests.
    """
    file_name = "01.jpg"

    return file_name


@pytest.fixture
def scores(scope="session"):
    """ This function defines the detected bounding boxes scores."""
    confidence_scores = np.array([0.6, 0.15], dtype=np.float)
    return confidence_scores


@pytest.fixture
def category_index(scope="session"):
    """ This function defines the category_index dictionary."""
    category_index = {0: {"id": 0, "name": "tunnel"}}
    return category_index


@pytest.fixture
def detection_model(experiment_config_file):
    """ This function defines the detection_model."""
    pipeline_config_path = experiment_config_file.training.model_config
    checkpoint_path = experiment_config_file.training.checkpoint_path
    num_classes = 90

    detection_model = restore_weights_from_checkpoint(
        pipeline_config_path=pipeline_config_path,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
    )
    return detection_model


@pytest.fixture
def detected_bboxes():
    """ This function defines two detected bounding boxes."""
    bboxes = np.array([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]])
    return bboxes


@pytest.fixture
def detected_text():
    """ This function defines two detected bounding boxes."""
    text_list = np.array(["text1", "text2"], dtype=np.str)
    return text_list


@pytest.fixture
def detected_image():
    """ This function defines one detected 512 x 512 image."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    return image


@pytest.fixture
def bboxes_normalized():
    """ This function defines the normalized bounding boxes."""
    bboxes = [[0.1, 0.3, 0.15, 0.35], [0.2, 0.4, 0.25, 0.45]]
    return bboxes


@pytest.fixture()
def groundtruths():
    """
    This function defines the groundtruth bounding boxes data that we will use during the tests.
    """
    groundtruth_data = [
        {"index": "0", "class_name": "tunnel", "score": 1.0, "bbox": (0, 0, 3, 6)},
        {"index": "1", "class_name": "tunnel", "score": 1.0, "bbox": (0, 0, 5, 8)},
    ]

    return groundtruth_data


@pytest.fixture()
def detections():
    """
    This function defines the detected bounding boxes data that we will use during the tests.
    """
    predicted_data = [
        {"index": "0", "class_name": "tunnel", "score": 1.0, "bbox": (0, 0, 3, 6)},
        {"index": "1", "class_name": "tunnel", "score": 1.0, "bbox": (0, 0, 5, 8)},
    ]

    return predicted_data


@pytest.fixture
def all_classes():
    """ This function defines the detected classes by our object detector."""
    classes = ["tunnel"]
    return classes
