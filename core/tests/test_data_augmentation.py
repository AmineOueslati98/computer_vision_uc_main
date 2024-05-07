""" This module contains the code tests for data augmentation."""
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
# pylint: disable=unused-import

import cv2
import numpy as np
import pytest
import tensorflow as tf
from scipy import ndimage

from core.data.data_augmentation import DataAugmentation
from core.tests.test_data_loader import config_file


@pytest.fixture
def groundtruth_text():
    """Ground truth fixture."""
    return "text"


@pytest.mark.parametrize("phase", ["train", "val", "eval"])
class TestDataAugmentation:
    """ This class includes the code tests for data augmentation."""

    @staticmethod
    def test_horizontal_flip(
        phase,
        input_image,
        bboxes_normalized,
        input_category_ids,
        config_file,
        groundtruth_text,
        input_file_name,
    ):
        """This function will test the horizontal_flip augmentation"""

        config_file.augmentation.horizontal_flip = 1
        config_file.augmentation.rotation = 0
        config_file.augmentation.random_brightness_contrast = 0

        data_augmentation = DataAugmentation(config=config_file, phase=phase)
        height, width, _ = config_file.data.input_shape
        aug_im, _, _, _, _ = data_augmentation.process_data(
            input_image,
            bboxes_normalized,
            input_category_ids,
            groundtruth_text,
            input_file_name,
        )
        aug_im = aug_im.numpy()
        if phase == "train":

            expected_output = cv2.flip(input_image, 1).astype(np.float32)
        else:
            expected_output = cv2.resize(
                input_image, dsize=(height, width), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        # Testing with a random photo from the batch

        assert np.array_equal(aug_im, expected_output)

    @staticmethod
    def test_rotation(
        phase,
        input_image,
        bboxes_normalized,
        input_category_ids,
        config_file,
        groundtruth_text,
        input_file_name,
    ):
        """This function will test the rotation augmentation"""

        config_file.augmentation.horizontal_flip = 0
        config_file.augmentation.rotation = 1
        config_file.augmentation.random_brightness_contrast = 0
        data_augmentation = DataAugmentation(config=config_file, phase=phase)
        height, width, _ = config_file.data.input_shape

        aug_im, _, _, _, _ = data_augmentation.process_data(
            input_image,
            bboxes_normalized,
            input_category_ids,
            groundtruth_text,
            input_file_name,
        )
        aug_im = aug_im.numpy()
        if phase == "train":

            expected_output = ndimage.rotate(input_image, 90).astype(np.float32)
        else:
            expected_output = cv2.resize(
                input_image, dsize=(height, width), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        assert np.array_equal(aug_im, expected_output)

    @staticmethod
    @pytest.mark.parametrize("expected_multiplier", [1])
    def test_random_brightness_contrast(
        phase,
        input_image,
        bboxes_normalized,
        expected_multiplier,
        input_category_ids,
        config_file,
        groundtruth_text,
        input_file_name,
    ):
        """This function will test the random_brightness_contrast augmentation"""

        config_file.augmentation.horizontal_flip = 0
        config_file.augmentation.rotation = 0
        config_file.augmentation.random_brightness_contrast = 1
        data_augmentation = DataAugmentation(config=config_file, phase=phase)

        height, width, _ = config_file.data.input_shape
        # Using the slice of input containing ones as an input to the data augmentation module
        ones_slice = input_image[:6, :, :]
        aug_im, _, _, _, _ = data_augmentation.process_data(
            ones_slice,
            bboxes_normalized,
            input_category_ids,
            groundtruth_text,
            input_file_name,
        )
        aug_im = aug_im.numpy()
        if phase == "train":

            expected_img = np.ones((6, 12, 3), dtype=np.float32) * expected_multiplier
        else:
            expected_img = cv2.resize(
                ones_slice, dsize=(height, width), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        assert np.array_equal(expected_img, aug_im)

    @staticmethod
    def test_data_augmentation(
        phase,
        input_image,
        bboxes_normalized,
        input_category_ids,
        config_file,
        groundtruth_text,
        input_file_name,
    ):
        """ This function will test the data types of the augmented datset """
        data_augmentation = DataAugmentation(config=config_file, phase=phase)

        (
            aug_im,
            aug_bboxes,
            aug_ids,
            groundtruth_text,
            file_name,
        ) = data_augmentation.process_data(
            input_image,
            bboxes_normalized,
            input_category_ids,
            groundtruth_text,
            input_file_name,
        )
        assert aug_im.dtype == tf.float32
        assert aug_bboxes.dtype == tf.float32
        assert aug_ids.dtype == tf.float32
        assert groundtruth_text.dtype == tf.string
        assert file_name.dtype == tf.string
