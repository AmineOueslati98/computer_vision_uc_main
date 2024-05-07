""" This module contains the code tests for the evaluation pipeline."""
# pylint: disable=redefined-outer-name

import logging
import math
import os
from io import StringIO

import numpy as np
import pytest
from object_detection.core.model import DetectionModel

from core.utils.configs import get_config_from_yaml
from core.utils.eval_utils import (
    Evaluator,
    convert_to_absolute_values,
    evaluate_ocr_result,
    filter_bboxes_by_score,
)
from core.utils.inference_utils import restore_weights_from_checkpoint
from core.workflows.evaluate import evaluation_pipeline

# from core.tests.conftest import test_env


def test_restore_weights_from_checkpoint(experiment_config_file):
    """ This function tests restore_weights_from_checkpoint function."""
    # pylint: disable=protected-access

    pipeline_config_path = experiment_config_file.training.model_config
    checkpoint_path = experiment_config_file.training.checkpoint_path
    num_classes = len(experiment_config_file.data.category_id_to_name)
    detection_model = restore_weights_from_checkpoint(
        pipeline_config_path=pipeline_config_path,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
    )

    assert isinstance(detection_model, DetectionModel)
    assert detection_model._num_classes == num_classes


def test_mean_average_precision(
    groundtruths, detections, all_classes, iou_threshold=0.5, is_training=False
):
    """ This function tests the mean_average_precision method from the Evaluator class."""

    evaluator = Evaluator()
    mean_average_precision = evaluator.mean_average_precision(
        groundtruths=groundtruths,
        detections=detections,
        classes=all_classes,
        iou_threshold=iou_threshold,
        is_training=is_training,
    )

    assert mean_average_precision == 1.0


def test_convert_to_absolute_values(bboxes_normalized):
    """ This function tests the convert_to_absolute_values function."""
    height, width = (320, 320)
    absolute_bboxes = []

    for bbox in bboxes_normalized:
        abs_bbox = convert_to_absolute_values(size=(height, width), bbox=bbox)
        absolute_bboxes.append(abs_bbox)

    expected_output = [(32, 96, 16, 16), (64, 128, 16, 16)]
    assert absolute_bboxes == expected_output


def test_filter_bboxes_by_score(
    bboxes_normalized, scores, input_category_ids, score_threshold=0.5
):
    """ This function tests the filter_bboxes_by_score function."""
    det_bboxes, det_scores, det_class_ids = filter_bboxes_by_score(
        det_bboxes=bboxes_normalized,
        det_scores=scores,
        det_class_ids=input_category_ids,
        score_threshold=score_threshold,
    )
    expected_bboxes = np.array([[0.1, 0.3, 0.15, 0.35]])
    expected_scores = np.array([0.6])
    expected_class_ids = np.array([0])
    assert np.array_equal(det_bboxes, expected_bboxes)
    assert np.array_equal(det_scores, expected_scores)
    assert np.array_equal(det_class_ids, expected_class_ids)


def test_evaluate_pipeline(experiment_config_file, test_dir, experiment_name, caplog):
    """ This function tests the evaluation pipeline."""
    # Include the whole created experiment name with the date
    for folder in os.listdir(test_dir):
        if experiment_name in folder:

            experiment_full_name = folder
    caplog.set_level(logging.INFO)

    evaluation_pipeline(
        experiment_path=os.path.join(test_dir, experiment_full_name),
        image_dir=os.path.join(test_dir, "test_images"),
        annot_file_path=os.path.join(test_dir, "test_annotations_json/test.json"),
        iou_threshold=0.5,
        score_threshold=0.01,
    )
    assert "mean average precision:" in caplog.text


def test_evaluate_ocr():
    """Test scoring string function."""
    ground_truth = "3km"
    close_result = "3kn"
    bad_result = "9 meters"
    perfect_score = evaluate_ocr_result(
        [ground_truth, ground_truth, ground_truth],
        [ground_truth, ground_truth, ground_truth],
    )
    assert math.isclose(perfect_score, 1.0)

    close_score = evaluate_ocr_result([ground_truth], [close_result])
    bad_score = evaluate_ocr_result([ground_truth], [bad_result])
    assert bad_score < close_score < 1
