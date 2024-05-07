"""SSD architecture fine tuning"""
import logging
from typing import List

import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch
from object_detection.utils import config_util

from core.models.models import ModelFineTune


class SsdFineTune(ModelFineTune):
    """SSD Fine tuning"""

    def restore_weights(self, model_config: str, checkpoint_path: str) -> SSDMetaArch:
        """
        Build custom model from the pipeline config and restore weights
        for all but last layer.
        args:
            model_config: the model configuration
            checkpoint_path: checkpoints of the pretrained model
        returns:
            detection_model: the built model.
        """
        # Load pipeline config and build the detection model.
        print("Building model and restoring weights for fine-tuning...", flush=True)
        configs = config_util.get_configs_from_pipeline_file(model_config)
        archi_config = configs["model"]
        archi_config.ssd.num_classes = self.num_classes
        archi_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=archi_config, is_training=True
        )
        # Set up object-based checkpoint restore --- We will
        # restore the box regression head but initialize
        # the classification head from scratch
        # pylint: disable=protected-access
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=(
                detection_model._box_predictor._base_tower_layers_for_heads
            ),
            _box_prediction_head=(detection_model._box_predictor._box_prediction_head),
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor,
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)

        # Restore the checkpoint to the checkpoint path
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        tmp_image, tmp_shapes = detection_model.preprocess(
            tf.zeros([1] + self.input_shape)
        )
        tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)
        _ = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)
        logging.info("Weights restored!")
        return detection_model

    def layers_to_fine_tune(self, detection_model: SSDMetaArch) -> List[tf.Variable]:
        """
        Select variables in top layers to fine-tune.
        args:
            detection_model: our model to fine tune
        returns:
            fine_tune_vars: list of variables to fine tune
        """

        trainable_variables = detection_model.trainable_variables
        fine_tune_vars = []
        prefixes_to_train = [
            "WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead",
            "WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead",
        ]
        for var in trainable_variables:
            for prefix in prefixes_to_train:
                if var.name.startswith(prefix):

                    fine_tune_vars.append(var)
                    break
        return fine_tune_vars
