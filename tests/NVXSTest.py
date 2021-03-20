import unittest
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.absolute())
from NVXS_Wrapper import NVXS_Wrapper as NVXS

import cv2

class NVXSTest(unittest.TestCase):

    def setUp(self):
        self.nvxs = NVXS()

    def test_get_faces_landmarks(self):
        """
            Test if NVXS can find the facial landmark features and return them in the proper structure.
            face-box: (,,,,)
            skin-color: (,,,)
            hair-color: (,,,)
            left-eye-box: (,,,,)
            right-eye-box: (,,,,)
            mouth-box: (,,,,)
            nose-pos: (,,)
            chin-pos: (,,)
        """
        def test(imagepath, expected_number_of_faces):
            img = cv2.imread(str(Path(__file__).parent.absolute() / imagepath))
            faces_landmarks = self.nvxs.get_faces_landmarks(img)
            self.assertEqual(len(faces_landmarks), expected_number_of_faces)

            landmark_structure = { 'face-box': 4, 'skin-color': 3,  'hair-color': 3, 'left-eye-box': 4, 'right-eye-box': 4, 'mouth-box': 4, 'nose-pos': 2, 'chin-pos': 2 }

            for landmarks in faces_landmarks:
                for key, size in landmark_structure.items():
                    self.assertIn(key, landmarks)
                    self.assertEqual(len(landmarks[key]), size)

        cases = [
            # single face images
            ('images/16_rot.jpg', 1),
            ('images/104.jpg', 1),
            ('images/295.jpg', 1),
            ('images/295.jpg', 1),
            ('images/moetron.jpg', 1),

            # multiple face images
            ('images/222.jpg', 2),
            ('images/158.jpg', 2),

            # no face images
            ('images/106.jpg', 0),
            ('images/132.jpg', 0),
            ('images/569.jpg', 0),
            ('images/899.jpg', 0)
        ]

        for case in cases:
            try:
                test(*case)
            except:
                print('Error:', case)
                raise

if __name__ == '__main__':
    unittest.main()
