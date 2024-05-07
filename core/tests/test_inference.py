""" This module contains the code tests for the inference pipeline."""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-import
# pylint: disable=too-many-arguments
import os

import matplotlib.pyplot as plt

from core.data.data_loader import get_dataset
from core.utils.inference_utils import get_model_detection_function, plot_detections
from core.workflows.inference import inference_pipeline


def test_plot_detections(
    experiment_config_file,
    test_dir,
    experiment_name,
    detected_image,
    detected_bboxes,
    input_category_ids,
    scores,
    category_index,
    detected_text,
):
    """ This function tests the plot_detection function from inference_utils."""
    for folder in os.listdir(test_dir):
        if experiment_name in folder:

            experiment_full_name = folder
    output_images_path = os.path.join(
        test_dir, experiment_full_name, "output_images/01.jpg"
    )
    plot_detections(
        image_np=detected_image,
        boxes=detected_bboxes,
        detected_text=detected_text,
        class_ids=input_category_ids,
        scores=scores,
        score_threshold=0.5,
        category_index=category_index,
    )
    assert plt.gcf().number == 1


def test_get_model_detection_function(
    detection_model, test_dir, experiment_config_file
):
    """ This function will test the detect_fn for the inference pipeline."""

    test_ds = get_dataset(
        config=experiment_config_file,
        phase="inference",
        image_dir=os.path.join(test_dir, "test_images"),
    )
    test_image_tensors, _ = next(iter(test_ds))

    detect_fn = get_model_detection_function(detection_model)

    post_processed_predictions = detect_fn(test_image_tensors)

    assert len(post_processed_predictions) == 8
    assert (
        len(post_processed_predictions["detection_boxes"])
        == len(post_processed_predictions["detection_classes"])
        == len(post_processed_predictions["detection_scores"])
    )


def test_inference_pipeline(experiment_config_file, test_dir, experiment_name):
    """ This function tests the inference pipeline."""
    # Include the whole created experiment name with the date
    for folder in os.listdir(test_dir):
        if experiment_name in folder:

            experiment_full_name = folder

    inference_pipeline(
        experiment_path=os.path.join(test_dir, experiment_full_name),
        image_dir=os.path.join(test_dir, "test_images"),
        score_threshold=0.01,
    )
    assert plt.gcf().number == 1
