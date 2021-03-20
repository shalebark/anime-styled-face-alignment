from aligner import Aligner
import cv2

class AlignerSingleton:
    __aligner = None

    @classmethod
    def instance(cls):
        if cls.__aligner is None:
            cls.__aligner = Aligner()

        return cls.__aligner

def get_and_align_all_faces(image=None, imagepath=None):
    """
        Returns an array of faces found in the image, rotated to align to the z-axis.

        Parameters:
            image: CV2 BGR Image
    """
    assert (image is not None) ^ (imagepath is not None)

    if imagepath:
        image = cv2.imread(imagepath)

    aligner = AlignerSingleton.instance()
    faces = aligner.get_faces_landmarks(image)
    return [aligner.align_and_extract_face(image.copy(), face) for face in faces]

