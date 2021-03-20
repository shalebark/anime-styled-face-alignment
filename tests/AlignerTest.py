import unittest
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.absolute())

from aligner import Aligner

import cv2
import numpy as np

class AlignerTest(unittest.TestCase):

    def setUp(self):
        self.aligner = Aligner()

    def test_align_and_extract_images(self):
        def test(imagepath, expected_number_of_faces, output_size, desired_left_eye_relative_position):
            img = cv2.imread(str(Path(__file__).parent.absolute() / 'images/' / imagepath))

            faces = self.aligner.get_faces_landmarks(img)
            self.assertEqual(len(faces), expected_number_of_faces)

            for i, face in enumerate(faces):
                output = self.aligner.align_and_extract_face(img.copy(), face, output_size, desired_left_eye_relative_position)

                t = Path(imagepath)
                numbered_output_imagepath = t.stem + '.' + str(i) + '.bmp'
                full_numbered_output_imagepath = str(Path(__file__).parent.absolute() / 'outputs/' / numbered_output_imagepath)
                test_output = cv2.imread(full_numbered_output_imagepath)

                # assert the output array is the same
                self.assertTrue(np.array_equal(test_output, output))

                # use this to generate to output
                # cv2.imwrite(full_numbered_output_imagepath, output)

        cases = [
            # single face images
            ('16_rot.jpg', 1, (128, 128), (.58, .58)),
            ('104.jpg', 1, (64, 64), (.58, .58)),
            ('295.jpg', 1, (240, 240), (.58, .58)),
            ('moetron.jpg', 1, (128, 3000), (.58, .58)),

            # # multiple face images
            ('222.jpg', 2, (128, 128), (.90, .58)),
            ('158.jpg', 2, (128, 128), (.58, .90)),

            # # no face images
            ('106.jpg', 0, (128, 128), (.58, .58)),
            ('132.jpg', 0, (128, 128), (.58, .58)),
            ('569.jpg', 0, (128, 128), (.58, .58)),
            ('899.jpg', 0, (128, 128), (.58, .58))
        ]

        for case in cases:
            try:
                test(*case)
            except:
                print('Error:', case)
                raise

if __name__ == '__main__':
    unittest.main()
