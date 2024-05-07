""" This module contains the inference utils """
# pylint: disable=too-many-arguments

import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from albumentations.augmentations.bbox_utils import denormalize_bboxes
from object_detection.builders import model_builder
from object_detection.core.model import DetectionModel
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils

from core.ocr.ocr import OcrModel
from core.utils.eval_utils import filter_bboxes_by_score


def plot_detections(
    image_np: np.ndarray,
    boxes: np.ndarray,
    detected_text: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    category_index: Dict[int, Dict[str, object]],
) -> np.ndarray:
    """Wrapper function to visualize detections.

    Args:
        image_np: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        detected_text: text detected by ocr
        class_ids: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
    Returns:
        image_np_with_annotations: the result image.

    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        class_ids,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=score_threshold,
    )
    if scores.any():
        boxes = boxes[scores > score_threshold]
        detected_text = detected_text[scores > score_threshold]
    denormalized_bboxes = np.asarray(
        denormalize_bboxes(boxes, image_np.shape[1], image_np.shape[0]), dtype=np.int32
    )
    for bbox_num, bbox in enumerate(denormalized_bboxes):
        bbox_text = detected_text[bbox_num]
        # Empty detections will be plotted as ""
        if not bbox_text:
            bbox_text = '""'
        cv2.putText(
            image_np_with_annotations,
            bbox_text,
            (bbox[1], bbox[2] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    return image_np_with_annotations


def get_model_detection_function(
    detection_model: DetectionModel,
) -> Callable[[List[tf.Tensor]], Dict[str, List[tf.Tensor]]]:
    """Get a tf.function for object detection.
    Args:
        detection_model: the detection model.
    Returns:
        detection_fn: tensorflow detection function.
    """

    @tf.function
    def detect_fn(input_tensor: List[tf.Tensor]) -> Dict[str, List[tf.Tensor]]:
        """Run detection on an input image.

        Args:
          input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
          post_processed_predictions: A dict containing 3 Tensors
          (`detection_boxes`, `detection_classes`,and `detection_scores`).
        """

        preprocessed_image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(preprocessed_image, shapes)
        post_processed_predictions = detection_model.postprocess(
            prediction_dict, shapes
        )
        return post_processed_predictions

    return detect_fn


def restore_weights_from_checkpoint(
    pipeline_config_path: str, checkpoint_path: str, num_classes: int
) -> DetectionModel:
    """This function restores weights from checkpoints and build a detection model.
    Args:
        pipeline_config_path: the configuration file containing all
        the model's architecture preferences.
        checkpoint_path: the path of the checkpoint to restore weights from.
        num_classes: the number of classes fed to the trained model.
    Returns:
        detection_model: the restored weights of the detection model.
    """
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs["model"]
    model_config.ssd.num_classes = num_classes
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()
    return detection_model


def infere_images(
    detection_model: DetectionModel,
    images: List[tf.Tensor],
    image_names: List[str],
    ocr: OcrModel,
    categories_dict: Dict[int, str],
    score_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """This function runs the object detection inference pipeline.
    Args:
        detection_model: the restored weights of the detection model.
        images: a list containing the test images.
        image_names: a list containing the image names.
        ocr: the ocr model.
        categories_dict: a dictionary mapping the categories to their indexes.
        score_threshold:the score threshold used to filter the bounding boxes.
    Returns:
        csv_data: a list containing each results data.
        output_images: images with detected bboxes and text
    """

    detect_fn = get_model_detection_function(detection_model)
    detections = detect_fn(images)

    label_id_offset = 0
    category_index = {}
    # Put the category_index in a specific format {id:{'id':id,'name':class_name}}
    # for our plotting function
    for class_id, class_name in categories_dict.items():

        category_index[class_id] = {"id": class_id, "name": class_name}

    logging.info("start inference!")
    csv_data = []
    output_images = []
    for i, _ in enumerate(images):
        det_bboxes = detections["detection_boxes"][i].numpy()
        det_scores = detections["detection_scores"][i].numpy()

        det_class_ids = detections["detection_classes"][i].numpy()
        det_bboxes, det_scores, det_class_ids = filter_bboxes_by_score(
            det_bboxes, det_scores, det_class_ids, score_threshold, False
        )
        image = images[i].numpy().astype(np.uint8)
        if det_bboxes.any():
            det_class_names = np.vectorize(categories_dict.get)(det_class_ids)
            denormalized_bboxes = np.asarray(
                denormalize_bboxes(det_bboxes, image.shape[1], image.shape[0]),
                dtype=np.int32,
            )
            detected_text = ocr(
                np.expand_dims(image, axis=0),
                None,
                np.expand_dims(denormalized_bboxes, axis=0),
                np.expand_dims(det_class_names, axis=0),
            )
            output_image = plot_detections(
                image,
                det_bboxes,
                np.array(detected_text, dtype=np.str),
                det_class_ids.astype(np.uint32) + label_id_offset,
                det_scores,
                score_threshold,
                category_index,
            )
            output_images.append(output_image)
        else:
            detected_text = np.array([], dtype=np.str)
            output_image = plot_detections(
                image,
                det_bboxes,
                detected_text,
                det_class_ids.astype(np.uint32) + label_id_offset,
                det_scores,
                score_threshold,
                category_index,
            )
            output_images.append(output_image)

        csv_columns = ["file_name", "predicted_boxes", "ocr_prediction"]
        csv_data.append(
            {
                "file_name": image_names[i],
                "predicted_boxes": det_bboxes,
                "ocr_prediction": detected_text,
            }
        )

    return csv_data, output_images
