"""Abstract fine tuning model.
This file defines a generic base class for fine tuning a detection model.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf
from object_detection.core.model import DetectionModel


class ModelFineTune(ABC):
    """ Abstract class for fine tuning a detection model"""

    def __init__(self, num_classes: int, input_shape: List[int]):
        """Constructor.
        Args:
        num_classes: number of classes.
        input_shape: the input shape of the model.
        """
        self.num_classes = num_classes
        self.input_shape = input_shape

    @abstractmethod
    def restore_weights(self, model_config: str, checkpoint_path: str):
        """
        Build custom model from the pipeline config and restore weights
        for all but last layer.
        args:
            model_config: the model configuration
            checkpoint_path: checkpoints of the pretrained model
        returns:
            detection_model: the built model.
        """

    @abstractmethod
    def layers_to_fine_tune(self, detection_model: DetectionModel):
        """
        Select variables in top layers to fine-tune.
        args:
            detection_model: our model to fine tune
        returns:
            fine_tune_vars: list of variables to fine tune
        """

    def __call__(
        self, model_config: str, checkpoint_path: str
    ) -> Tuple[DetectionModel, List[tf.Variable]]:
        """
        This method calls the restore_weights and layers_to_fine_tune functions
        sequentially and returns their output.
        Args:
            model_config: the model configuration
            checkpoint_path: checkpoints of the pretrained model
        Returns:
            detection_model: the built model.
            ckpt: model restored checkpoints.
            fine_tune_vars: list of variables to fine tune.

        """
        detection_model = self.restore_weights(model_config, checkpoint_path)
        fine_tune_vars = self.layers_to_fine_tune(detection_model)
        return detection_model, fine_tune_vars
