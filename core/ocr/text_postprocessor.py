import abc
import re

from core.ocr.config import SignOcrPostprocessorConfig


class TextPostprocessor(metaclass=abc.ABCMeta):
    """Abstract Class for text postprocessing."""

    @abc.abstractmethod
    def __call__(self, txt: str) -> str:
        """PostProcessing abstract function.
        args:
            txt: string that we want to process.
        returns:
            a processed string.
        """
        pass


class SignOcrPostprocessor(TextPostprocessor):
    """Class for Processing texts obtained from OCR : for Sign Text Recognition."""

    def __init__(self, config: SignOcrPostprocessorConfig) -> None:
        super().__init__()
        self.config = config

    def __call__(self, txt: str):
        """Extract a valid distance string from any string
        args:
            txt: string that we want to extract a valid distance from it.
        returns:
            extracted_distance_string: distance string that would be a number
                followed by m or km or an empty string.
        """
        distance_string = txt.lower()
        for item in self.config.pre_regular_expression:
            distance_string = re.sub(item[0], item[1], distance_string)
        grammar = "".join(self.config.grammar)
        distance_string_group = re.search(grammar, distance_string)
        if distance_string_group:
            distance_string = distance_string_group.group(0)
            # if the string contains k we change it directely to km
            if distance_string.find("k") != -1:
                distance_string = re.sub("k.*", "km", distance_string)
            else:
                # the string does not contain k. We determine with the distance if it is m or km
                # We use only the integer part to determine the distance
                integer_digits_group = re.search(
                    self.config.grammar[0], distance_string
                )
                float_digits_group = re.search(
                    "".join(self.config.grammar[:-1]), distance_string
                )
                if integer_digits_group and float_digits_group:
                    distance_integer_part = int(integer_digits_group.group(0))
                    if distance_integer_part < self.config.shortest_tunnel_length:

                        distance_string = float_digits_group.group(0) + "km"
                    else:
                        distance_string = float_digits_group.group(0) + "m"
            # we change . by , as it is the convention in the signs

            for item in self.config.post_regular_expression:
                distance_string = re.sub(item[0], item[1], distance_string)
            return distance_string
        return ""
