###########################################################################################
#                                                                                         #
# Evaluator class: Implements the most popular metrics for object detection               #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################
""" This module contains the evaluation utils for object detection models."""
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements


import os
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from albumentations.augmentations.bbox_utils import denormalize_bbox

from core.utils.configs import ExperimentConfig


class Evaluator:
    """This class is used to calculate the mean average precision metric to evaluate
    our model's performance."""

    @staticmethod
    def mean_average_precision(
        groundtruths: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        classes: List[str],
        iou_threshold: float = 0.5,
        is_training: bool = True,
    ) -> float:
        """Calculate the mean average precision according to metrics used by the
         VOC Pascal 2012 challenge.

        Args:
            groundtruths: a list representing groundtruth bounding
            boxes organized in the following format
            [bounding box index,class_name,confidence,(bb coordinates XYWH)].
            detections: a list representing detected bounding boxes organized in the
            following format [bounding box index,class_name,confidence,(bb coordinates XYWH)]
            classes: a list representing all classes.
            iou_threshold: IOU threshold indicating which detections will be considered
            as a true positive or false positive.
            (default value = 0.5).
            is_training: a boolean indincating whether we are using this method in a
            training phase or not.
        Returns:
            mean_average_precision: the mean average precision between all classes.
        """

        classes = sorted(classes)

        cumulative_average_precision = 0.0
        valid_classes = 0

        for class_name in classes:
            # Get only detection of class c
            detections_per_class = [
                detection
                for detection in detections
                if detection["class_name"] == class_name
            ]
            # Get only ground truths of class c
            groundtruths_per_class = [
                groundtruth
                for groundtruth in groundtruths
                if groundtruth["class_name"] == class_name
            ]

            total_positives = len(groundtruths_per_class)
            # sort detections by decreasing confidence
            detections_per_class = sorted(
                detections_per_class,
                key=lambda detection: detection["score"],
                reverse=True,
            )
            true_positives = np.zeros(len(detections_per_class))
            false_positives = np.zeros(len(detections_per_class))
            # create dictionary with amount of groundtruths for each image
            groundtruths_per_image = Counter(
                [groundtruth["index"] for groundtruth in groundtruths_per_class]
            )
            for image_index, number_of_groundtruths in groundtruths_per_image.items():
                groundtruths_per_image[image_index] = np.zeros(number_of_groundtruths)

            # Loop through detections
            for detection in detections_per_class:
                # Find ground truth image
                groundtruth_per_detection = [
                    groundtruth
                    for groundtruth in groundtruths_per_class
                    if groundtruth["index"] == detection["index"]
                ]
                iou_max = sys.float_info.min
                for j, _ in enumerate(groundtruth_per_detection):
                    # calculate the Intersection over union between the
                    # groundtruth bbox and detected bbox
                    iou = Evaluator.intersection_over_union(
                        groundtruth_per_detection[j]["bbox"], detection["bbox"]
                    )
                    if iou > iou_max:
                        iou_max = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iou_max >= iou_threshold:
                    bbox_index = detection["index"]
                    assigned_detection: np.ndarray = groundtruths_per_image[bbox_index]

                    if assigned_detection[jmax] == 0:
                        true_positives[
                            detections_per_class.index(detection)
                        ] = 1  # count the detected box as true positive
                    # flag as already 'seen'

                    assigned_detection[jmax] = 1

                else:
                    false_positives[
                        detections_per_class.index(detection)
                    ] = 1  # count the detected box as false positive
                # compute precision, recall and average precision

            cumulative_false_positives = np.cumsum(false_positives)

            cumulative_true_positives = np.cumsum(true_positives)
            recall = cumulative_true_positives / total_positives
            precision = np.divide(
                cumulative_true_positives,
                (cumulative_false_positives + cumulative_true_positives),
            )

            # compute the average precision, interpolated precision and interpolated recall
            (
                average_precision,
                interpolated_precision,
                interpolated_recall,
            ) = Evaluator.calculate_average_precision(recall, precision)

            if total_positives > 0:
                valid_classes += 1
                cumulative_average_precision += average_precision
                precision = ["%.2f" % p for p in precision]
                recall = ["%.2f" % r for r in recall]
                ap_str = "{0:.5f}".format(average_precision)

            if not is_training:
                mean_average_precision = cumulative_average_precision / valid_classes

                output_file = open(os.path.join("results.txt"), "w")
                output_file.write("Object Detection Metrics\n")
                output_file.write(
                    "Average Precision (AP), Precision and Recall per class:"
                )
                output_file.write("\n\nClass: %s" % class_name)
                output_file.write("\nAP: %s" % ap_str)
                output_file.write("\nPrecision: %s" % precision)
                output_file.write(
                    "\nInterpolated Precision: %s" % interpolated_precision
                )
                output_file.write("\nInterpolated Recall: %s" % interpolated_recall)

                output_file.write("\nRecall: %s" % recall)
                output_file.write(
                    "\n\n\nmean_average_precision: %s"
                    % "{0:.5f}".format(mean_average_precision)
                )

                output_file.close()
            else:

                mean_average_precision = cumulative_average_precision / valid_classes

        return mean_average_precision

    @staticmethod
    def calculate_average_precision(
        recall: List[float], precision: List[float]
    ) -> Tuple[float, List[float], List[float]]:
        """This method calculates the average precision for each class using
         all points interpolation.

        Args:
            recall: a list containing the recall for all the detected bounding
             boxes.
            precision: a list containing the precision for all the detected
            bounding boxes.
        Returns:
            average_precision: the average precision for each class.
            interpolated_recall: a list containing the interpolated recall for
             all detections.
            interpolated_precision: a list containing the interpolated precision
            for all detections.
            segment_indexes: a list containing the indexes for
            AUC (Area under the ROC Curve) segments used to calculate the average precision.

        """
        interpolated_recall = [0.0] + list(recall) + [1.0]

        interpolated_precision = [0.0] + list(precision) + [0.0]

        # Calculating the AUC(Area under curve AUC) of the precision x recall curve
        for i in range(len(interpolated_precision) - 1, 0, -1):
            interpolated_precision[i - 1] = max(
                interpolated_precision[i - 1], interpolated_precision[i]
            )
        # Defining the segments of the AUC
        segment_indexes = []
        for i in range(len(interpolated_recall) - 1):
            if interpolated_recall[1:][i] != interpolated_recall[0:-1][i]:
                segment_indexes.append(i + 1)

        average_precision = 0
        # Calculate the average precision using the AUC segments

        for index in segment_indexes:
            average_precision = average_precision + np.sum(
                (interpolated_recall[index] - interpolated_recall[index - 1])
                * interpolated_precision[index]
            )
        interpolated_precision = interpolated_precision[0:-1]
        interpolated_recall = interpolated_recall[0 : len(interpolated_precision) - 1]
        return (average_precision, interpolated_precision, interpolated_recall)

    @staticmethod
    def intersection_over_union(
        groundtruth_bbox: Tuple[int, int, int, int],
        predicted_bbox: Tuple[int, int, int, int],
    ) -> float:
        """This method is used to calculate the intersection over union between the
         overlapping groundtruth and detected bounding boxes.
        Args:
            groundtruth_bbox: the groundtruth bounding box.
            predicted_bbox: the detected bounding box.
        Returns:
            iou: the intersection over union score.
        """

        inter_box_top_left = [
            max(groundtruth_bbox[0], predicted_bbox[0]),
            max(groundtruth_bbox[1], predicted_bbox[1]),
        ]
        inter_box_bottom_right = [
            min(
                groundtruth_bbox[0] + groundtruth_bbox[2],
                predicted_bbox[0] + predicted_bbox[2],
            ),
            min(
                groundtruth_bbox[1] + groundtruth_bbox[3],
                predicted_bbox[1] + predicted_bbox[3],
            ),
        ]

        inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
        inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

        intersection = inter_box_w * inter_box_h
        union = (
            groundtruth_bbox[2] * groundtruth_bbox[3]
            + predicted_bbox[2] * predicted_bbox[3]
            - intersection
        )
        if union:
            iou = intersection / union
        else:
            iou = 0

        return iou


def convert_to_absolute_values(
    size: Tuple[int, int], bbox: List[float]
) -> Tuple[int, int, int, int]:
    """This function is used to convert the bounding boxes from relative format with the
     image size to the absolute format.

    Args:
        size: a tuple defining the image size (height,width).
        bbox: a bounding in the relative format .
    Returns:
        abs_bbox: the bounding box converted to the absolute format (x,y,w,h).
    """
    denormalized_bbox = tuple(
        round(coordinate) for coordinate in denormalize_bbox(bbox, size[1], size[0])
    )
    # change from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height)
    abs_bbox = (
        denormalized_bbox[0],
        denormalized_bbox[1],
        denormalized_bbox[2] - denormalized_bbox[0],
        denormalized_bbox[3] - denormalized_bbox[1],
    )
    return abs_bbox


def get_groundtruths_and_detections(
    config: ExperimentConfig,
    image_tensors: List[tf.Tensor],
    groundtruth_boxes_list: List[tf.Tensor],
    groundtruth_encoded_class_ids: List[tf.Tensor],
    processed_dict: Dict[str, List[tf.Tensor]],
    score_threshold: float = 0.5,
    is_training: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], float, float]:

    """This function is used to prepare the groundtruths and detections for the Evaluator.
    Args:
        config: the configuration file containing all the preferences.
        image_tensors: a list of input image tensors fed to our model.
        groundtruth_boxes_list: a list containing the groundtruth bounding boxes tensors.
        groundtruth_encoded_class_ids: a list of one hot encoded class ids tensors.
        processed_dict: a dictionary containing the postprocessed model's predictions.
        score_threshold: the score threshold used to filter the bounding boxes.
        is_training: a boolean specifing whether we are using this function in
         the training or the evalutation phase.
    Returns:
        groundtruths: a list representing groundtruth bounding boxes organized in the
         following format [bounding box index,class_name,confidence,(bb coordinates XYWH)].
        detections: a list representing detected bounding boxes organized in the
        following format [bounding box index,class_name,confidence,(bb coordinates XYWH)]
        classes: a list representing all classes.
        end_det: a variable indicating the end of detection time.
        fps: a variable indicating the frames per second for the detection."""

    groundtruths = []
    detections = []

    classes = []
    categories_dict = config.data.category_id_to_name

    image_index = 0
    start_det = time.time()

    for i in range(len(image_tensors)):

        label_indexes = tf.argmax(groundtruth_encoded_class_ids[i], axis=1).numpy()
        for bbox, index in zip(groundtruth_boxes_list[i], label_indexes):
            bbox = bbox.numpy()
            class_name = categories_dict[index]
            # Convert bboxes from (ymin,xmin,ymax,xmax) to (xmin,ymin,xmax,ymax)
            converted_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            # Convert bboxes from (xmin,ymin,xmax,ymax) to (xcenter,ycenter,width,height)

            abs_bbox = convert_to_absolute_values(
                size=(image_tensors[i].shape[0], image_tensors[i].shape[1]),
                bbox=converted_bbox,
            )
            groundtruth_bbox_data = {
                "index": str(image_index),
                "class_name": class_name,
                "score": 1.0,
                "bbox": abs_bbox,
            }
            groundtruths.append(groundtruth_bbox_data)

        det_bboxes = processed_dict["detection_boxes"][i].numpy()
        det_scores = processed_dict["detection_scores"][i].numpy()

        det_class_ids = processed_dict["detection_classes"][i].numpy()

        if not is_training:
            # the filtering function will filter bboxes by a certain
            # score and Convert bboxes from (ymin,xmin,ymax,xmax) to
            # (xmin,ymin,xmax,ymax)

            det_bboxes, det_scores, det_class_ids = filter_bboxes_by_score(
                det_bboxes, det_scores, det_class_ids, score_threshold, True
            )

        else:
            # Convert bboxes from (ymin,xmin,ymax,xmax) to (xmin,ymin,xmax,ymax)
            det_bboxes = [[bbox[1], bbox[0], bbox[3], bbox[2]] for bbox in det_bboxes]
        det_bboxes = np.array(det_bboxes)

        for det_bbox, det_score, det_class_id in zip(
            det_bboxes, det_scores, det_class_ids
        ):
            # Convert bboxes from (xmin,ymin,xmax,ymax) to (xcenter,ycenter,width,height)
            abs_bbox = convert_to_absolute_values(
                size=(image_tensors[i].shape[0], image_tensors[i].shape[1]),
                bbox=det_bbox,
            )
            detected_bbox_data = {
                "index": str(image_index),
                "class_name": categories_dict[det_class_id],
                "score": det_score,
                "bbox": abs_bbox,
            }

            detections.append(detected_bbox_data)
            classes.append(categories_dict[det_class_id])

        image_index += 1
    classes.sort()
    end_det = time.time() - start_det
    fps = len(image_tensors) / end_det

    return groundtruths, detections, classes, end_det, fps


def filter_bboxes_by_score(
    det_bboxes: np.ndarray,
    det_scores: np.ndarray,
    det_class_ids: np.ndarray,
    score_threshold: float = 0.5,
    width_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function keeps only the  detected bboxes
    (as well as the corresponding scores and class_ids) with score
    above the threshold.
    Args:
        det_bboxes: the predicted bounding boxes.
        det_scores: the predicted scores.
        det_class_ids: the predicted class ids.
        score_threshold:the score threshold used to filter the bounding boxes.
        width_first: a boolean specifing whether we will invert the bounding box
         coordinates or not (ymin,xmin,ymax,xmax) --> (xmin,ymin,xmax,ymax)
    Returns:
        det_bboxes: the filtered bounding boxes.
        det_scores: the filtered scores.
        det_class_ids: the filtered class ids.
    """
    det_bboxes = np.squeeze(det_bboxes)
    det_scores = np.squeeze(det_scores)
    det_class_ids = np.squeeze(det_class_ids)
    res = np.where(det_scores > score_threshold)
    if not res[0].shape[0]:
        det_bboxes = np.zeros((0, 4))
        det_scores = np.zeros((0, 1))
        det_class_ids = np.zeros((0, 1))
        return det_bboxes, det_scores, det_class_ids
    filtering_index = np.where(det_scores > score_threshold)[0][-1] + 1

    # this creates an array with just enough rows as object with score above the threshold
    if width_first:
        # format: absolute x, y, x, y
        det_bboxes = np.array(
            [
                det_bboxes[:filtering_index, 1],
                det_bboxes[:filtering_index, 0],
                det_bboxes[:filtering_index, 3],
                det_bboxes[:filtering_index, 2],
            ]
        ).T
    else:
        det_bboxes = np.array(
            [
                det_bboxes[:filtering_index, 0],
                det_bboxes[:filtering_index, 1],
                det_bboxes[:filtering_index, 2],
                det_bboxes[:filtering_index, 3],
            ]
        ).T
    det_class_ids = det_class_ids[:filtering_index]
    det_scores = det_scores[:filtering_index]
    return det_bboxes, det_scores, det_class_ids


def evaluate_ocr_result(ground_truth_list: List[str], detected_list: List[str]):
    """Compare between two strings and return the result.
    Args:
        ground_truth_list: a list containing groundtruth text values for all the images.
        detected_list: a list containing predicted text values for all the images.
    Returns:
        score: evaluation score for the ocr.

    """
    score = [
        SequenceMatcher(None, ground_truth, ocr_result).ratio()
        for ground_truth, ocr_result in zip(ground_truth_list, detected_list)
    ]
    return sum(score) / len(score)
