"""Test the changes made to ppgan realsr_predictor"""

from hashlib import sha1

import cv2
import numpy as np
import py
import pytest
from numpy.core.fromnumeric import shape
from ppgan.apps import RealSRPredictor


@pytest.fixture
def super_resolution():
    return RealSRPredictor(output="")


@pytest.fixture
def rgb_images():
    """Input images in the RGB format"""
    image = np.array(
        [
            [[198, 233, 229], [27, 168, 149], [198, 233, 229], [255, 255, 255]],
            [[138, 182, 156], [33, 152, 155], [122, 182, 187], [140, 179, 193]],
            [[198, 233, 229], [27, 168, 149], [65, 161, 161], [122, 182, 187]],
            [[255, 255, 255], [138, 182, 156], [195, 204, 182], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    return np.array([image, image], dtype=np.uint8)


@pytest.fixture
def normalized_images():
    """Input images changed to channel first and normalized to [0, 1]"""
    normalized_image = np.array(
        [
            [
                [0.7764706, 0.10588235, 0.7764706, 1.0],
                [0.5411765, 0.12941177, 0.47843137, 0.54901963],
                [0.7764706, 0.10588235, 0.25490198, 0.47843137],
                [1.0, 0.5411765, 0.7647059, 1.0],
            ],
            [
                [0.9137255, 0.65882355, 0.9137255, 1.0],
                [0.7137255, 0.59607846, 0.7137255, 0.7019608],
                [0.9137255, 0.65882355, 0.6313726, 0.7137255],
                [1.0, 0.7137255, 0.8, 1.0],
            ],
            [
                [0.8980392, 0.58431375, 0.8980392, 1.0],
                [0.6117647, 0.60784316, 0.73333335, 0.75686276],
                [0.8980392, 0.58431375, 0.6313726, 0.73333335],
                [1.0, 0.6117647, 0.7137255, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    return np.array([normalized_image, normalized_image])


def test_norm_images(super_resolution, rgb_images, normalized_images):
    normalized_images_output = super_resolution.norm_images(rgb_images)
    assert np.allclose(normalized_images, normalized_images_output)


def test_denorm_images(super_resolution, normalized_images, rgb_images):
    denormalized_images_output = super_resolution.denorm_images(normalized_images)
    assert np.allclose(rgb_images, denormalized_images_output, rtol=1)
