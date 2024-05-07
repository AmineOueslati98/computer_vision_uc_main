"""Config for OCR"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from dataclasses_json import DataClassJsonMixin


class ThresholdingMode(Enum):
    """Describes how to perform the optimization."""

    Binary = "binary"
    Otsu = "otsu"
    AdaptiveGaussian = "adaptive_gaussian"


@dataclass
class OcrConfig(DataClassJsonMixin):
    """OCR preprocessing and model config."""

    preprocessing: "OcrPreprocessingConfig"
    model: "OcrModelConfig"
    postprocessing: "SignOcrPostprocessorConfig"


@dataclass
class OcrModelConfig(DataClassJsonMixin):
    """OCR model config."""

    engine: str = "paddle"  # paddle  #easy
    paddle_pretrained: str = ""  # paddle  #easy
    gpu_memory: int = 6000
    rec_image_shape: str = "3, 32, 320"
    rec_batch_num: int = 6
    max_text_length: int = 10
    use_space_char: bool = False
    drop_score: float = 0.5
    # post process result text. The text will be normalized to distances eg:3,1km, 1000m..
    post_process: bool = True
    detect_box: bool = True


@dataclass
class SignOcrPostprocessorConfig(DataClassJsonMixin):
    """Sign OCR Post Processsor Config."""

    shortest_tunnel_length: int = 100
    pre_regular_expression: List[List[str]] = field(
        default_factory=lambda: [["[;,:]+", "."], ["[^0-9km.]+", ""]]
    )
    grammar: List[str] = field(default_factory=lambda: ["[0-9]+", "\.?[0-9]*", "k?m?"])
    post_regular_expression: List[List[str]] = field(
        default_factory=lambda: [["\.", ","]]
    )


# pylint: disable=too-many-instance-attributes
@dataclass
class OcrPreprocessingConfig(DataClassJsonMixin):
    """OCR config."""

    # box coordinates are given as follow [left, upper, right, lower] with each value in [0,1]
    # e.g: [0, 0, 1, 1] will return the full image without cropping
    crop_box_coordinate: List[float] = field(
        default_factory=lambda: [
            0.15401514524219048,
            0.6203247251311627,
            0.8828889841714775,
            0.9658599127976383,
        ]
    )
    # desired height and width of the cropped image
    # The image does not change its size if you put a 0 in height or width
    cropped_part_size: List[int] = field(default_factory=lambda: [200, 100])
    # change the perspective of an image to make it straight.
    apply_deskewing: bool = True
    # hsv value range of the desired part to deskew. This is used to make a better edge
    # detection of the object to deskew. Use opencv to determine the best ranges for hsv.
    # Only when apply_deskewing those values are processed.
    # The values are for h,s and v: {category_name: [lower_range, upper_range]}
    h_range_for_deskewing: Dict[str, List[int]] = field(
        default_factory=lambda: {"tunnel": [90, 120]}
    )
    s_range_for_deskewing: Dict[str, List[int]] = field(
        default_factory=lambda: {"tunnel": [200, 255]}
    )
    v_range_for_deskewing: Dict[str, List[int]] = field(
        default_factory=lambda: {"tunnel": [60, 255]}
    )
    # use ppgan to make the resolution of the image better.
    apply_super_resolution_gan: bool = True
    super_resolution_batch_size: int = 4

    apply_adding_border: bool = False
    # border box are given as follow [left, upper, right, lower] with each value
    #  the number of padded pixels
    border_box: List[float] = field(default_factory=lambda: [10, 10, 190, 90])

    apply_histogram_equilization: bool = False
    clip_limit: int = 1
    tile_grid_size: List[int] = field(default_factory=lambda: [4, 4])
    # remove noise.
    apply_median_blur: bool = False
    median_blur_kernel_size: int = 3
    # change brightness and contrast.
    apply_brightness_contrast: bool = True
    brightness: int = 160
    contrast: int = 110

    apply_thresholding: bool = False
    # available modes are: binary, otsu and adaptive_gaussian
    threshold_mode: ThresholdingMode = ThresholdingMode.AdaptiveGaussian
    # variables for adaptive_gaussian threshold
    adaptive_threshold_bloc_size: int = 35
    adaptive_threshold_constant: float = 2
    # variable for binary and otsu threshold.
    threshold: int = 127

    apply_inverting: bool = True

    apply_dilation: bool = False
    dilation_kernel_size: List[int] = field(default_factory=lambda: [2, 2])
    dilation_number_of_iterations: int = 1

    apply_erosion: bool = False
    erosion_kernel_size: List[int] = field(default_factory=lambda: [2, 2])
    erosion_number_of_iterations: int = 1
