"""This module runs the custom training loop"""
# pylint: disable=no-member
# pylint: disable=E1129
import datetime
import logging
import os

import tensorflow as tf

from core.models.ssd_fine_tune import SsdFineTune
from core.utils.eval_utils import Evaluator
from core.utils.train_utils import (
    from_tensor_to_list,
    get_model_train_step_function,
    get_model_val_step_function,
    get_training_config,
)


# custom training
def train(config_file_path: str):
    """
    Run the eager custom training
    Args:
        config_file_path: configuration file containing information about the data and
            training hyperparameters
    """
    # pylint: disable=too-many-locals
    # the number is reasonable in our case.
    (
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
    ) = get_training_config(config_file_path)

    # Create model and restore weights for all but last layer
    if model_name == "ssd":
        model_to_fine_tune = SsdFineTune(num_classes, input_shape)
    else:
        raise RuntimeError("The model architecture must be either SSD or EfficientDet")
    detection_model, layers_to_fine_tune = model_to_fine_tune(
        model_config, checkpoint_path
    )

    tf.keras.backend.set_learning_phase(True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    evaluator = Evaluator()
    train_step_fn = get_model_train_step_function(
        config_file, detection_model, optimizer, evaluator, layers_to_fine_tune
    )
    val_step_fn = get_model_val_step_function(config_file, detection_model, evaluator)
    # Create the checkpoint objects
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model, step=tf.Variable(1))
    manager = tf.train.CheckpointManager(ckpt, output_model_path, max_to_keep=None)
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(logs_path, current_time, "train")
    val_log_dir = os.path.join(logs_path, current_time, "val")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    # run training
    print("Start fine-tuning!", flush=True)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    for idx in range(num_batches):
        image_tensors_train, gt_boxes_train, gt_classes_train, _, _ = next(
            iter(train_ds)
        )
        image_tensors_val, gt_boxes_val, gt_classes_val, _, _ = next(iter(val_ds))
        groundtruth_boxes_list_train = from_tensor_to_list(gt_boxes_train)
        groundtruth_classes_list_train = from_tensor_to_list(gt_classes_train)
        groundtruth_boxes_list_val = from_tensor_to_list(gt_boxes_val)
        groundtruth_classes_list_val = from_tensor_to_list(gt_classes_val)
        # Training step (forward pass + backwards pass)
        train_total_loss, train_metrics = train_step_fn(
            image_tensors_train,
            groundtruth_boxes_list_train,
            groundtruth_classes_list_train,
        )
        val_total_loss, val_metrics = val_step_fn(
            image_tensors_val, groundtruth_boxes_list_val, groundtruth_classes_list_val
        )
        # checkpointing
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 5 == 0:
            save_path = manager.save()
            logging.info(
                "Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path)
            )

        if idx % 5 == 0:
            # write tensorboard logs
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_total_loss.numpy(), step=idx)
                tf.summary.scalar("mean_average_precision", train_metrics, step=idx)
            with val_summary_writer.as_default():
                tf.summary.scalar("loss", val_total_loss.numpy(), step=idx)
                tf.summary.scalar("mean_average_precision", val_metrics, step=idx)

            print(
                "batch "
                + str(idx)
                + " of "
                + str(num_batches)
                + ", train_loss="
                + str(train_total_loss.numpy())
                + ", val_loss="
                + str(val_total_loss.numpy())
                + ", mean_average_precision_train="
                + str(train_metrics)
                + ", mean_average_precision_val="
                + str(val_metrics),
                flush=True,
            )

    logging.info("Done fine-tuning!")
