import abc
from collections.abc import Iterable
from typing import Tuple, List

class FaceLandmarkDetectorBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect_landmarks(self, image, facebox: Tuple[int, int, int, int]) -> dict:
        """
            Returns the an dictionary of facial landmarks found in the face.
            Dict fields:
                    * left/right from the character's perspective
                    * pos fields are points (Tuple[int, int])
                    'face-box' # this should be the same as the facebox passed in.
                    'face-right-pos'
                    'chin-pos'
                    'face-left-pos'
                    'right-brow-right-pos'
                    'right-brow-middle-pos'
                    'right-brow-left-pos'
                    'left-brow-right-pos'
                    'left-brow-middle-pos'
                    'left-brow-left-pos'
                    'nose-pos'
                    'right-eye-top-right-pos'
                    'right-eye-top-middle-pos'
                    'right-eye-top-left-pos'
                    'right-eye-bottom-pos'
                    'right-eye-center-pos'
                    'left-eye-top-right-pos'
                    'left-eye-top-middle-pos'
                    'left-eye-top-left-pos'
                    'left-eye-bottom-pos'
                    'left-eye-center-pos'
                    'mouth-top-right-pos'
                    'mouth-top-middle-pos'
                    'mouth-top-left-pos'
                    'mouth-bottom-pos'
                    'mouth-center-pos'
                    'left-eye-color'
                    'right-eye-color'
                    'hair-color'
            Parameters:
                image: The image where faces are detected from. CV2 BGR Image
                facebox: A tuple containing the bounding box on the face. Tuple(x,y,w,h)
        """