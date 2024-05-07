"""Test functions for core.ocr.image_preprocessing"""
# pylint: disable=redefined-outer-name
import os
from importlib import resources

import cv2
import numpy as np
import pytest
from paddleocr import PaddleOCR

from core.ocr.config import OcrPreprocessingConfig, ThresholdingMode
from core.ocr.image_preprocessing import (
    create_preprocessed_image_batch,
    preprocess_image,
)


@pytest.fixture()
def preprocessing_input_image():
    """ This function defines the input image that we will use during the tests."""
    image_array_1 = np.ones((4, 8, 3), dtype=np.uint8)
    image_array_2 = np.ones((4, 8, 3), dtype=np.uint8) * 2
    input_image = np.vstack((image_array_1, image_array_2))

    return input_image


@pytest.fixture()
def image_class():
    return "tunnel"


@pytest.fixture()
def ocr_config():
    """Ocr config fixture """
    return OcrPreprocessingConfig(
        crop_box_coordinate=[0, 0, 1, 1],
        cropped_part_size=[8, 8],
        apply_deskewing=False,
        apply_adding_border=False,
        apply_brightness_contrast=False,
        apply_inverting=False,
        apply_histogram_equilization=False,
        apply_median_blur=False,
        apply_thresholding=False,
        apply_erosion=False,
        apply_dilation=False,
        apply_super_resolution_gan=False,
    )


def test_invert(preprocessing_input_image, image_class, ocr_config):
    """Test inverting."""
    ocr_config.apply_inverting = True
    result_image = np.vstack(
        (np.full((4, 8), 254, dtype=np.uint8), np.full((4, 8), 253, dtype=np.uint8))
    )
    preprocessed_image = preprocess_image(
        preprocessing_input_image, image_class, ocr_config
    )
    assert np.array_equal(preprocessed_image, result_image)


def test_basic_preprocessing(preprocessing_input_image, image_class, ocr_config):
    """Test resizing and changing it to grayscale."""
    ocr_config.cropped_part_size = [4, 4]
    result_image = np.vstack(
        (np.ones((2, 4), dtype=np.uint8), 2 * np.ones((2, 4), dtype=np.uint8))
    )

    preprocessed_image = preprocess_image(
        preprocessing_input_image, image_class, ocr_config
    )
    assert np.array_equal(preprocessed_image, result_image)


def test_cropping(preprocessing_input_image, image_class, ocr_config):
    """Test cropping inside the bounding box area."""
    # No image resizing
    ocr_config.cropped_part_size = (0, 0)
    ocr_config.crop_box_coordinate = (0.25, 0.25, 0.75, 0.5)
    result_image = np.ones((2, 4), dtype=np.uint8)
    preprocessed_image = preprocess_image(
        preprocessing_input_image, image_class, ocr_config
    )
    assert np.array_equal(preprocessed_image, result_image)


def test_adding_border(preprocessing_input_image, image_class, ocr_config):
    """Test add border to image."""
    ocr_config.apply_adding_border = True
    ocr_config.border_box = [10, 30, 15, 20]
    grayscale_input_image = np.vstack(
        (np.ones((4, 8), dtype=np.uint8), np.full((4, 8), 2, dtype=np.uint8))
    )
    middle_result_image_part = np.hstack(
        (
            np.full((8, 10), 255, dtype=np.uint8),
            grayscale_input_image,
            np.full((8, 15), 255, dtype=np.uint8),
        )
    )
    result_image = np.vstack(
        (
            np.full((30, 33), 255, dtype=np.uint8),
            middle_result_image_part,
            np.full((20, 33), 255, dtype=np.uint8),
        )
    )
    preprocessed_image = preprocess_image(
        preprocessing_input_image, image_class, ocr_config
    )
    assert np.array_equal(preprocessed_image, result_image)


@pytest.mark.parametrize("threshold_mode,", ["binary", "adaptive_gaussian", "otsu"])
def test_thresholding(
    preprocessing_input_image, image_class, ocr_config, threshold_mode
):
    """Test thresholding the image."""
    ocr_config.apply_thresholding = True
    ocr_config.threshold_mode = ThresholdingMode(threshold_mode)
    ocr_config.adaptive_threshold_bloc_size = 21
    ocr_config.adaptive_threshold_constant = 1
    ocr_config.threshold = 200
    modified_input_image = np.array(preprocessing_input_image, dtype=np.uint8)
    modified_input_image[5, 5] = [230, 230, 230]
    result_image = np.zeros((8, 8), dtype=np.uint8)
    result_image[5, 5] = 255
    preprocessed_image = preprocess_image(modified_input_image, image_class, ocr_config)
    assert np.array_equal(preprocessed_image, result_image)


def test_create_preprocessed_image_batch(ocr_config):
    """Test preprocessing a batch of images.
    For this test the bbox input is the same in the two formats
    accepted by the preprocessing function. We check the output is the same for both formats."""
    image_name = "01.jpg"
    with resources.path("core.tests.fixtures", image_name) as image_path:
        ocr_test_image = cv2.imread(str(image_path))
        ocr_test_image = cv2.cvtColor(ocr_test_image, cv2.COLOR_BGR2RGB)
    ocr_test_image = np.array([ocr_test_image, ocr_test_image])
    category_classes = np.array([["tunnel", "tunnel"], ["tunnel", "tunnel"]])
    bbox_inputs = [
        np.array(
            [
                [[315, 868, 418, 915], [0, 0, 20, 20]],
                [[10, 10, 50, 50], [20, 20, 60, 60]],
            ]
        ),
        [
            {"index": "0", "bbox": (315, 868, 418, 915), "class_name": "tunnel"},
            {"index": "0", "bbox": (0, 0, 20, 20), "class_name": "tunnel"},
            {"index": "1", "bbox": (10, 10, 50, 50), "class_name": "tunnel"},
            {"index": "1", "bbox": (20, 20, 60, 60), "class_name": "tunnel"},
        ],
    ]
    results = []
    for bbox in bbox_inputs:
        if isinstance(bbox, list):
            preprocessed_images = create_preprocessed_image_batch(
                ocr_test_image, bbox, None, None, ocr_config
            )
        else:
            preprocessed_images = create_preprocessed_image_batch(
                ocr_test_image, None, bbox, category_classes, ocr_config
            )
        results.append(preprocessed_images)

        assert len(preprocessed_images) == 4
        # check that all the created images were resized to (8,8) and changed to gray format.
        for image in preprocessed_images:
            assert image.shape == (8, 8)

    # check that both outputs(for the 2 bbox types) are identical.
    for preprocessed_image1, preprocessed_image2 in zip(results[0], results[0]):
        assert np.array_equal(preprocessed_image1, preprocessed_image2)


def test_preprocess_for_ocr():
    """Apply all the preprocessing techniques on a real image and
    evaluate their effectiveness on an ocr example.
    This function calls all the preprocessing techniques except
    of super resolution to reduce computation time."""
    complete_ocr_config = OcrPreprocessingConfig(
        crop_box_coordinate=[0.19, 0.70, 0.82, 0.94],
        cropped_part_size=[200, 80],
        apply_deskewing=True,
        h_range_for_deskewing={"blue_sign": [90, 120], "white_sign": [0, 255]},
        s_range_for_deskewing={"blue_sign": [200, 255], "white_sign": [0, 255]},
        v_range_for_deskewing={"blue_sign": [60, 255], "white_sign": [70, 165]},
        apply_adding_border=False,
        apply_histogram_equilization=True,
        clip_limit=1,
        tile_grid_size=[2, 2],
        apply_median_blur=True,
        median_blur_kernel_size=3,
        apply_brightness_contrast=True,
        brightness=180,
        contrast=120,
        apply_thresholding=False,
        apply_inverting=True,
        apply_dilation=True,
        dilation_kernel_size=[2, 2],
        dilation_number_of_iterations=1,
        apply_erosion=True,
        erosion_kernel_size=[2, 2],
        erosion_number_of_iterations=1,
        apply_super_resolution_gan=True,
    )
    image_list = ["01.jpg", "48.jpg"]
    ocr_results = ["3km", "2000m"]
    images = []
    for image_name in image_list:
        with resources.path("core.tests.fixtures", image_name) as image_path:
            ocr_test_image = cv2.imread(str(image_path))
            ocr_test_image = cv2.cvtColor(ocr_test_image, cv2.COLOR_BGR2RGB)
            images.append(ocr_test_image)

    ocr_test_image = np.array(images)
    images_bounding_boxes = np.array([[[315, 868, 418, 915]], [[414, 1100, 487, 1235]]])
    image_class = np.array([["blue_sign"], ["white_sign"]])

    preprocessed_image = create_preprocessed_image_batch(
        ocr_test_image, None, images_bounding_boxes, image_class, complete_ocr_config
    )
    ocr = PaddleOCR(lang="en")
    for image, ocr_groundtruth in zip(preprocessed_image, ocr_results):
        result = ocr.ocr(image, det=False)[0][0].strip()
        assert result == ocr_groundtruth
