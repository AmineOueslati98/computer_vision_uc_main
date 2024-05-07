"""Helper function for the training pipeline"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=no-name-in-module


from typing import Callable, List, Tuple

import tensorflow as tf
from object_detection.core.model import DetectionModel
from tensorflow.python.data.ops.dataset_ops import Dataset

from core.data.data_loader import get_dataset
from core.utils.configs import ExperimentConfig, get_config_from_yaml
from core.utils.eval_utils import Evaluator, get_groundtruths_and_detections


# Set up forward + backward pass for a single train step.
def get_model_train_step_function(
    config_file: ExperimentConfig,
    model: DetectionModel,
    optimizer: tf.keras.optimizers,
    evaluator: Evaluator,
    vars_to_fine_tune: List[tf.Variable],
) -> Callable[
    [List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Tuple[tf.Tensor, float]
]:
    """
    Get a tf.function for training step.
    Args:
        config_file: the configuration file containing all the preferences.
        model: detection model
        optimizer: optimizer for updating weights
        vars_to_fine_tune: list of variables to fine tune
        inmput_shape: input shape of the model
    Returns:
        validation step function
    """

    # @tf.function
    def train_step_fn(
        image_tensors: List[tf.Tensor],
        groundtruth_boxes_list: List[tf.Tensor],
        groundtruth_classes_list: List[tf.Tensor],
    ) -> Tuple[tf.Tensor, float]:
        """
        A single training iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list,
        )

        with tf.GradientTape() as tape:
            preprocessed_images, shapes = model.preprocess(image_tensors)

            prediction_dict = model.predict(preprocessed_images, shapes)
            processed_dict = model.postprocess(prediction_dict, shapes)

            gt_bboxes, dt_bboxes, all_classes, _, _ = get_groundtruths_and_detections(
                config_file,
                image_tensors,
                groundtruth_boxes_list,
                groundtruth_classes_list,
                processed_dict,
            )

            metrics = evaluator.mean_average_precision(
                gt_bboxes, dt_bboxes, all_classes
            )
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = (
                losses_dict["Loss/localization_loss"]
                + losses_dict["Loss/classification_loss"]
            )
            # train_acc_metric.update_state(preprocessed_images, prediction_dict)

            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss, metrics

    return train_step_fn


def get_model_val_step_function(
    config_file: ExperimentConfig, model: DetectionModel, evaluator
) -> Callable[
    [List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Tuple[tf.Tensor, float]
]:
    """
    Get a tf.function for val step.
    Args:
        config_file: the configuration file containing all the preferences.

        model: detection model
        inmput_shape: input shape of the model
    Returns:
        validation step function
    """

    # @tf.function
    def val_step_fn(
        image_tensors: tf.Tensor,
        groundtruth_boxes_list: List[tf.Tensor],
        groundtruth_classes_list: List[tf.Tensor],
    ) -> Tuple[tf.Tensor, float]:
        """
        A single val iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list,
        )
        preprocessed_images, shapes = model.preprocess(image_tensors)
        prediction_dict = model.predict(preprocessed_images, shapes)
        processed_dict = model.postprocess(prediction_dict, shapes)
        gt_bboxes, dt_bboxes, all_classes, _, _ = get_groundtruths_and_detections(
            config_file,
            image_tensors,
            groundtruth_boxes_list,
            groundtruth_classes_list,
            processed_dict,
        )

        metrics = evaluator.mean_average_precision(gt_bboxes, dt_bboxes, all_classes)

        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = (
            losses_dict["Loss/localization_loss"]
            + losses_dict["Loss/classification_loss"]
        )
        return total_loss, metrics

    return val_step_fn


def from_tensor_to_list(ground_truth_tf: tf.Tensor) -> List[tf.Tensor]:
    """
    Convert a tensor to list of tensors after removing the padding.
    args:
        ground_truth_tf: ground truth tensor that we want to convert
    returns:
        ground_truth_list: list of ground truth tensors
    """
    ground_truth_list = []
    for ground_truth_component in ground_truth_tf:
        padding = tf.math.reduce_all(ground_truth_component == 0, axis=1)
        ground_truth_list.append(
            tf.gather_nd(ground_truth_component, tf.where(~padding))
        )
    return ground_truth_list


def get_training_config(
    config_file_path: str,
) -> Tuple[
    ExperimentConfig,
    Dataset,
    Dataset,
    List[int],
    int,
    str,
    str,
    str,
    str,
    str,
    float,
    int,
]:
    """
    Load training parameters from the configuration file.
    args:
        config_file_path: configuration file containing information about the data and
            training hyperparameters.
    returns:
        config_file: the configuration file containing all the preferences.

        train_ds: loaded training dataset,
        val_ds: loaded validation dataset,
        input_shape: the input shape of the model,
        num_batches: number of iterations,
        model_config: path of the podel configuration,
        output_model_path: path of the output model checkpoints,
        model_name: the used object detection architecture,
        checkpoint_path: the path of the pretrained model checkpoints,
        logs_path: path of tensorboard logs,
        learning_rate: the learning rate of the optimizer,
        num_classes: number of classes to detect.

    """
    config_file = get_config_from_yaml(config_file_path)
    # load data
    train_ds = get_dataset(config_file, phase="train")
    val_ds = get_dataset(config_file, phase="val")
    label_dict = config_file.data.category_id_to_name
    input_shape = config_file.data.input_shape
    # load training hyperparameters from the config file
    num_batches = config_file.training.num_batches
    model_config = config_file.training.model_config
    output_model_path = config_file.training.output_model_path
    model_name = config_file.training.model_name.split("_")[0]
    checkpoint_path = config_file.training.checkpoint_path
    logs_path = config_file.training.logs_path
    learning_rate = config_file.training.learning_rate
    num_classes = len(label_dict)
    return (
        config_file,
        train_ds,
        val_ds,
        input_shape,
        num_batches,
        model_config,
        output_model_path,
        model_name,
        checkpoint_path,
        logs_path,
        learning_rate,
        num_classes,
    )
