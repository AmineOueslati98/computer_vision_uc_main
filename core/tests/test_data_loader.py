""" This module contains the code tests for data loader."""
import os
from dataclasses import asdict

import numpy as np
import pytest
import tensorflow as tf

# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
import yaml
from albumentations.augmentations.bbox_utils import normalize_bbox

from core.data.data_loader import DataLoader, get_dataset
from core.utils.configs import get_config_from_yaml


@pytest.fixture()
def config_file(test_env, test_dir):
    """ This function defines the config file."""
    with open(test_dir.join("config.yaml"), "w") as cfg:
        config_dict = asdict(test_env)
        config_dict["ocr"]["preprocessing"]["threshold_mode"] = config_dict["ocr"][
            "preprocessing"
        ]["threshold_mode"].value
        yaml.dump(config_dict, cfg)
    config = get_config_from_yaml(test_dir.join("config.yaml"))
    return config


@pytest.mark.parametrize("phase", ["train", "val", "eval"])
class TestDataLoader:
    """ This class includes the code tests for data loader."""

    @staticmethod
    def test_data_generator(
        config_file, test_dir, phase, input_image, input_bboxes, input_category_ids
    ):
        """ This function tests the data generator of the data loader."""
        data_loader = DataLoader(
            config=config_file,
            phase=phase,
            image_dir_test=os.path.join(test_dir, "test_images"),
            annot_file_path_test=os.path.join(
                test_dir, "test_annotations_json/test.json"
            ),
        )
        image, bboxes, category_ids, _, _ = next(data_loader.data_generator())
        height, width, _ = input_image.shape

        expected_bboxes = [
            list(normalize_bbox(bbox, height, width)) for bbox in input_bboxes
        ]
        assert np.array_equal(input_image, image)
        assert np.array_equal(expected_bboxes, bboxes)
        assert np.array_equal(input_category_ids, category_ids)

    @staticmethod
    def test_get_dataset(test_dir, phase, config_file):
        """This function will test the image and annotation files formats and the batch size
        of the output dataset"""

        output_dataset = get_dataset(
            config=config_file,
            phase=phase,
            image_dir=os.path.join(test_dir, "test_images"),
            annot_file_path=os.path.join(test_dir, "test_annotations_json/test.json"),
        )
        aug_im, aug_bboxes, aug_ids, ground_truth, file_name = next(
            iter(output_dataset)
        )

        assert (
            aug_im.shape[0] == config_file.training.batch_size
        ), "the batching was not done properly"
        assert aug_im.dtype == tf.float32
        assert aug_bboxes.dtype == tf.float32
        assert aug_ids.dtype == tf.float32
        assert ground_truth.dtype == tf.string
        assert file_name.dtype == tf.string
