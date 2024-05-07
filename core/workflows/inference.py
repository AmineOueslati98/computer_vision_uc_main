""" This module runs the model inference from a specific checkpoint."""
import csv
import logging

# pylint: disable=too-many-locals
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from albumentations.augmentations.bbox_utils import denormalize_bboxes

from core.data.data_loader import get_dataset
from core.ocr.ocr import OcrModel
from core.utils.configs import ExperimentConfig, get_config_from_yaml
from core.utils.eval_utils import filter_bboxes_by_score
from core.utils.inference_utils import (
    get_model_detection_function,
    infere_images,
    plot_detections,
    restore_weights_from_checkpoint,
)


def inference_pipeline(
    experiment_path: str, image_dir: str, score_threshold: float = 0.5
) -> None:
    """This function runs the object detection inference pipeline.
    Args:
        experiment_path: the path of the experiment to use in inference.
        image_dir: the directory containing the test images.
        score_threshold:the score threshold used to filter the bounding boxes.

    """
    start = time.time()
    experiment_config_path = os.path.join(experiment_path, "config.yaml")
    config = get_config_from_yaml(experiment_config_path)
    pipeline_config = config.training.model_config

    num_classes = len(config.data.category_id_to_name)
    # Set the batch_size equal to the number of test images to load all the images
    config.training.batch_size = len(os.listdir(image_dir))
    checkpoint_dir = os.path.join(experiment_path, "model_checkpoints")
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
    detection_model = restore_weights_from_checkpoint(
        pipeline_config, checkpoint, num_classes
    )
    test_ds = get_dataset(config=config, phase="inference", image_dir=image_dir)
    (test_image_tensors, file_names) = next(iter(test_ds))
    file_names = [
        file_names[file_name_num].numpy().decode("utf8")
        for file_name_num in range(file_names.shape[0])
    ]
    categories_dict = config.data.category_id_to_name

    output_dir_path = os.path.join(experiment_path, "output_images")

    logging.info("start inference!")
    start_inf = time.time()
    ocr = OcrModel(config.ocr)
    csv_data, output_images = infere_images(
        detection_model=detection_model,
        images=test_image_tensors,
        image_names=file_names,
        ocr=ocr,
        categories_dict=categories_dict,
        score_threshold=score_threshold,
    )
    for image, image_name in zip(output_images, file_names):
        output_path = os.path.join(output_dir_path, image_name)

        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, output_image)
    csv_file = os.path.join(experiment_path, "output.csv")
    try:
        with open(csv_file, "w") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["file_name", "predicted_boxes", "ocr_prediction"]
            )
            writer.writeheader()
            for row in csv_data:

                writer.writerow(row)
    except IOError:
        logging.info("Saving output file failed!")
    end_inf = time.time() - start_inf
    fps = len(test_image_tensors) / end_inf
    end = time.time()
    logging.info(
        "Elapsed time in detection was {:2.3f} secs at {:2.3f} fps".format(end_inf, fps)
    )
    logging.info("Total elapsed time in inference was {:2.3f} secs".format(end - start))
