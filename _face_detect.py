# using the animeface library, library supports facial feature locations
import animeface
import _util as util

def face_detect(img):
    """
        Returns an animeface object.
        If more than one face detected, only the largest and most plausible face will be used.
        Returns one face object. If no face objects found, returns false. 

        Parameters:
            img: cv2 image
    """
    faces = animeface.detect(util.convert_cv2_to_pil_image(img))
    if (len(faces) == 0):
        return False
    return faces[0]



