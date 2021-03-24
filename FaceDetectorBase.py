import abc
from collections.abc import Iterable
from typing import Tuple, List

class FaceDetectorBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect_faceboxes(self, image) -> List[Tuple[int, int, int, int]]:
        """
            Returns the an array of boxes containing each face.
            The box should be in (x,y,w,h) form.

            Parameters:
                image: The image where faces are detected from. CV2 BGR Image
        """