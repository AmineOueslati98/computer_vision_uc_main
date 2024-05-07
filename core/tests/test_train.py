# pylint: disable=redefined-outer-name
# pylint: disable=no-member
"""Test of the train module"""
import os
from dataclasses import asdict
from io import StringIO

import tensorflow as tf
import yaml

from core.utils import exp_utils, train_utils
from core.workflows import train


def test_from_tensor_to_list():
    """
    Tests the conversion of a tensor to a list of tensors.
    Tests the removing of the padding from the ground truth functions.
    """
    test_tensor1 = tf.concat(
        [tf.random.normal([1, 4], 0, 1, tf.float32), tf.zeros([2, 4])], 0
    )
    test_tensor2 = tf.concat(
        [tf.random.normal([2, 4], 0, 1, tf.float32), tf.zeros([1, 4])], 0
    )
    groundtruth_tensor = tf.convert_to_tensor([test_tensor1, test_tensor2])
    list_tf = train_utils.from_tensor_to_list(groundtruth_tensor)
    assert isinstance(list_tf, list)
    assert tf.is_tensor(list_tf[0])
    assert tf.is_tensor(list_tf[1])
    assert list_tf[0].shape == [1, 4]
    assert list_tf[1].shape == [2, 4]


def test_train(experiment_config_file, test_dir, experiment_name):
    """
    Tests the train function: verifies if the checkpoints are saved
    and tensorboard logs are generated.
    """

    new_experiment_name = [
        fold for fold in os.listdir(test_dir) if fold.startswith(experiment_name)
    ][0]
    assert (
        len(
            os.listdir(os.path.join(test_dir, new_experiment_name, "model_checkpoints"))
        )
        != 0
    )
    assert len(os.listdir(os.path.join(test_dir, new_experiment_name, "logs"))) != 0
