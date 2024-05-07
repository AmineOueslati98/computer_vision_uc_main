""" This module contains the code tests for data visualization."""
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
# pylint: disable=unused-import

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

from core.data.data_visualization import Visualization
from core.tests.test_data_loader import config_file


def input_generator():
    """ This function defines the input_generator that we will use during the tests."""

    for _ in range(10):

        input_image = np.zeros((512, 512, 3), dtype=np.uint8)
        bboxes = [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]]

        category_ids = [0, 0]
        one_hot_encoded_ids = tf.one_hot(
            indices=category_ids, depth=1, on_value=None, off_value=None
        )
        text_groundtruth = "500 m"
        file_name = "01.jpg"

        yield input_image, bboxes, one_hot_encoded_ids, text_groundtruth, file_name


@pytest.fixture()
def input_dataset(config_file):
    """ This function defines the input dataset that we will use during the tests."""
    dataset = tf.data.Dataset.from_generator(
        generator=input_generator,
        output_types=(tf.uint8, tf.float32, tf.float32, tf.string, tf.string),
    )
    batched_dataset = dataset.batch(batch_size=config_file.training.batch_size)
    return batched_dataset


@pytest.mark.parametrize("phase", ["train", "val", "eval"])
class TestVisualization:
    """ This class contains the test functions of data visualization."""

    @staticmethod
    def test_view_image(input_dataset, phase, config_file):
        """ This function will test view_image."""
        plt.close()
        visualizer = Visualization(config=config_file, phase=phase)
        visualizer.view_image(dataset=input_dataset, block=False)
        assert plt.gcf().number == 1

    @staticmethod
    def test_visualize_annotated_objects(input_dataset, phase, config_file):
        """ This function will test visualize_annotated_objects."""
        plt.close()
        visualizer = Visualization(config=config_file, phase=phase)
        visualizer.visualize_annotated_objects(dataset=input_dataset, block=False)
        assert plt.gcf().number == 1
