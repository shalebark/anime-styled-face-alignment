import PIL.Image as Image
import animeface
import numpy as np

class NVXS_Wrapper:

    def get_faces_landmarks(self, image):
        pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
        faces = animeface.detect(pil_image)

        landmarks = []
        for face in faces:
            landmarks.append({
                'face-box': tuple(face.face.pos.__dict__.values()),
                'skin-color': tuple(face.skin.color.__dict__.values()),
                'hair-color': tuple(face.hair.color.__dict__.values()),
                'left-eye-box': tuple(face.left_eye.pos.__dict__.values()),
                'right-eye-box': tuple(face.right_eye.pos.__dict__.values()),
                'mouth-box': tuple(face.mouth.pos.__dict__.values()),
                'nose-pos': tuple(face.nose.pos.__dict__.values()),
                'chin-pos': tuple(face.chin.pos.__dict__.values()),
            })
        return landmarks
