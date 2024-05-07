"""OCR class tests."""
# pylint: disable=redefined-outer-name
# pylint: disable=no-name-in-module

from importlib import resources

import cv2
import numpy as np
import pytest

from core.ocr.config import (
    OcrConfig,
    OcrModelConfig,
    OcrPreprocessingConfig,
    SignOcrPostprocessorConfig,
)
from core.ocr.ocr import OcrModel
from core.ocr.text_postprocessor import SignOcrPostprocessor


@pytest.fixture()
def ocr_config():
    """Ocr config fixture."""
    return OcrConfig(
        preprocessing=OcrPreprocessingConfig(
            crop_box_coordinate=(0.19, 0.70, 0.82, 0.94),
            cropped_part_size=(200, 80),
            apply_deskewing=True,
            h_range_for_deskewing={"tunnel": [90, 120]},
            s_range_for_deskewing={"tunnel": [200, 255]},
            v_range_for_deskewing={"tunnel": [60, 255]},
            apply_adding_border=False,
            apply_brightness_contrast=True,
            apply_inverting=True,
            apply_histogram_equilization=False,
            apply_median_blur=False,
            apply_thresholding=False,
            apply_erosion=False,
            apply_dilation=False,
            apply_super_resolution_gan=False,
        ),
        model=OcrModelConfig(
            engine="paddle",
            gpu_memory=6000,
            rec_image_shape="3, 32, 320",
            rec_batch_num=6,
            max_text_length=10,
            use_space_char=False,
            drop_score=0.5,
            post_process=True,
            detect_box=False,
        ),
        postprocessing=SignOcrPostprocessorConfig(
            shortest_tunnel_length=100,
            pre_regular_expression=[("[;,:]+", "."), ("[^0-9km.]+", "")],
            grammar=["[0-9]+", "\.?[0-9]*", "k?m?"],
            post_regular_expression=[("\.", ",")],
        ),
    )


@pytest.mark.parametrize(
    "input_string,expected_string",
    [
        ("3,6km", "3,6km"),
        ("3.6km", "3,6km"),
        ("3,6", "3,6km"),
        ("3,6kmm", "3,6km"),
        ("3,6bm", "3,6km"),
        ("kl3,6knj", "3,6km"),
        ("1550", "1550m"),
    ],
)
def test_postprocess(ocr_config, input_string, expected_string):
    """Test postprocessing distance strings."""
    Postprocessor = SignOcrPostprocessor(ocr_config.postprocessing)
    result = Postprocessor(input_string)
    assert expected_string == result


@pytest.mark.parametrize(
    "input_string,grammar,expected_string",
    [
        ("3,6km", ["[0-9]+", "\.?[0-9]*", "k?m?"], "3,6km"),
        ("03.6km", ["[1-9][0-9]*", "\.?[0-9]*", "k?m?"], "3,6km"),
        ("03.06785km", ["[1-9][0-9]*", "\.?[0-9]{0,3}", "k?m?"], "3,067km"),
    ],
)
def test_postprocess_grammar(ocr_config, input_string, grammar, expected_string):
    """Test postprocessing distance strings."""
    config = ocr_config.postprocessing
    config.grammar = grammar
    Postprocessor = SignOcrPostprocessor(config)
    result = Postprocessor(input_string)
    assert expected_string == result


@pytest.mark.parametrize(
    "detect_box,cropped_part_size", [(False, (200, 80)), (True, (200, 400))]
)
def test_ocr_model(ocr_config, detect_box, cropped_part_size):
    """Apply all the preprocessing techniques on a real image and
    evaluate their effectiveness on an ocr example.
    This function calls all the preprocessing techniques except
    of super resolution to reduce computation time."""
    ocr_config.model.detect_box = detect_box
    ocr_config.preprocessing.cropped_part_size = cropped_part_size
    with resources.path("core.tests.fixtures", "01.jpg") as image_path:
        ocr_test_image = cv2.imread(str(image_path))
        ocr_test_image = cv2.cvtColor(ocr_test_image, cv2.COLOR_BGR2RGB)
    batch_image = np.array([ocr_test_image])
    image_info = {
        "index": "0",
        "class_name": "tunnel",
        "score": 1.0,
        "bbox": (315, 868, 418, 915),
    }

    image_info = [image_info, image_info]
    ocr_model = OcrModel(ocr_config)
    result = ocr_model(batch_image, image_info, None, None)

    assert result == ["3km", "3km"]


def test_image_with_multiple_text_lines(ocr_config):
    """Try ocr on an image with multiple lines out of grammar and only one line following the grammar."""
    ocr_config.model.detect_box = True
    ocr_config.preprocessing.apply_deskewing = False
    ocr_config.preprocessing.apply_inverting = False
    with resources.path("core.tests.fixtures", "48.jpg") as image_path:
        ocr_test_image = cv2.imread(str(image_path))
        ocr_test_image = cv2.cvtColor(ocr_test_image, cv2.COLOR_BGR2RGB)

    batch_image = np.array([ocr_test_image])
    image_info = {
        "index": "0",
        "class_name": "tunnel",
        "score": 1.0,
        "bbox": (414, 1100, 487, 1235),
    }
    image_info = [image_info]
    ocr_model = OcrModel(ocr_config)
    result = ocr_model(batch_image, image_info, None, None)

    assert result == ["2000m"]
