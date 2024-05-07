""" This module runs the evaluation pipeline to determine the trained models performance"""
# pylint: disable=too-many-locals

import logging
import os
import time

import tensorflow as tf

from core.data.data_loader import get_dataset
from core.ocr.ocr import OcrModel
from core.utils.configs import ExperimentConfig, get_config_from_yaml
from core.utils.eval_utils import (
    Evaluator,
    evaluate_ocr_result,
    get_groundtruths_and_detections,
)
from core.utils.inference_utils import restore_weights_from_checkpoint


def evaluation_pipeline(
    experiment_path: str,
    image_dir: str,
    annot_file_path: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> None:
    """This function contains the evaluation pipeline to get the model's performance
     from any checkpoint.
    Args:
        experiment_path: the path of the experiment to evaluate.
        image_dir: the directory containing the test images.
        annot_file_path: the annotation data path.
        iou_threshold: the threshold to apply the intersection over union.
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
    test_ds = get_dataset(
        config=config,
        phase="eval",
        image_dir=image_dir,
        annot_file_path=annot_file_path,
    )

    (
        test_image_tensors,
        test_groundtruth_bboxes,
        test_groundtruth_encoded_class_ids,
        test_groundtruth_text,
        _,
    ) = next(iter(test_ds))

    preprocessed_image, shapes = detection_model.preprocess(test_image_tensors)

    prediction_dict = detection_model.predict(preprocessed_image, shapes)

    processed_dict = detection_model.postprocess(prediction_dict, shapes)
    logging.info("start evaluating!")

    (
        groundtruth_bboxes,
        detected_bboxes,
        all_classes,
        det_time,
        fps,
    ) = get_groundtruths_and_detections(
        config=config,
        image_tensors=test_image_tensors,
        groundtruth_boxes_list=test_groundtruth_bboxes,
        groundtruth_encoded_class_ids=test_groundtruth_encoded_class_ids,
        processed_dict=processed_dict,
        is_training=False,
        score_threshold=score_threshold,
    )
    all_classes = list(all_classes)
    all_classes.sort()

    evaluator = Evaluator()

    mean_average_precision = evaluator.mean_average_precision(
        groundtruths=groundtruth_bboxes,
        detections=detected_bboxes,
        classes=all_classes,
        iou_threshold=iou_threshold,
        is_training=False,
    )
    # change the bbox from [xmin, ymin, width, height] to [ymin, xmin, ymax, xmax]
    for bbox_info in detected_bboxes:
        bbox_info["bbox"] = (
            bbox_info["bbox"][1],
            bbox_info["bbox"][0],
            bbox_info["bbox"][1] + bbox_info["bbox"][3],
            bbox_info["bbox"][0] + bbox_info["bbox"][2],
        )
    ocr = OcrModel(config.ocr)
    ocr_result = ocr(test_image_tensors.numpy(), detected_bboxes, None, None)
    ocr_score = evaluate_ocr_result(test_groundtruth_text.numpy().flatten(), ocr_result)

    logging.info("mean average precision: {0:.5f}".format(mean_average_precision))
    logging.info("OCR score: {0:.5f}".format(ocr_score))
    end = time.time()
    logging.info(
        "Elapsed time in detection was {:2.3f} secs at {:2.3f} fps".format(
            det_time, fps
        )
    )
    logging.info(
        "Total elapsed time in evaluation was {:2.3f} secs".format(end - start)
    )
