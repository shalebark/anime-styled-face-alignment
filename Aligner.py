# from NVXS_Wrapper import NVXS_Wrapper as NVXS
# from CFA_Wrapper import CFA_Wrapper as CFA
import cv2
import numpy as np
import warnings

import os
from pathlib import Path
os.sys.path.append(Path(__file__).resolve().parent.__str__())

class Aligner:

    # get the angle needed to unrotate the face. angle is in radians
    def determine_rotation_angle(self, landmarks):
        """
            Returns the z-angle tiled by the face.

            Parameters:
                landmarks: An individual faces landmarks returns from get_faces_landmarks
        """
        lp = landmarks['left-eye-center-pos']
        rp = landmarks['right-eye-center-pos']
        return angle_between_points(lp, rp)

    def determine_bounding_box_of_rotated_box(self, box, rotation_matrix):
        """
            Returns a bounding box, that will contain the box after its' been transformed by the rotation_matrix

            Parameters:
                box: The box before it's been transformed.
                rotation_matrix: The matrix that will transform the box.
        """

        # top left, top right, bottom left, bottom right
        p1, p2, p3, p4 = box_points(box)

        # rotate all the points of the box
        tp1 = calc_rotate_point_with_rotation_matrix(p1, rotation_matrix)
        tp2 = calc_rotate_point_with_rotation_matrix(p2, rotation_matrix)
        tp3 = calc_rotate_point_with_rotation_matrix(p3, rotation_matrix)
        tp4 = calc_rotate_point_with_rotation_matrix(p4, rotation_matrix)

        # figure out which point has the furthest x distance, and the furthest y distance
        dx1 = abs(tp1[0] - tp4[0])
        dx2 = abs(tp2[0] - tp3[0])
        dy1 = abs(tp1[1] - tp4[1])
        dy2 = abs(tp2[1] - tp3[1])
        # the width and the height is the max distance between x and y
        w, h = max(dx1, dx2), max(dy1, dy2)

        # x and y is the min x, and min y among all points
        x = min(tp1[0], tp2[0], tp3[0], tp4[0])
        y = min(tp1[1], tp2[1], tp3[1], tp4[1])

        return (x, y, w, h)

    def determine_rotation_matrix(self, origin, angle, scale):
        """
            Returns the rotation matrix needed to rotate an image by origin, at angle, and then scale.

            Parameters:
                origin: The point of origin. (tuple)
                angle: The degree of rotation. (radians)
                scale: The scale. (float)
        """
        # scaling will be ignored at this step
        rotation_matrix = cv2.getRotationMatrix2D(origin, angle * 180 / np.pi, scale)
        return rotation_matrix

    def rotate_image_to_alignment(self, image, rotation_matrix, output_img_box):
        """
            Returns the rotated image, and the image will be correctly bounded so that the entire rotated image fits inside the output image.

            Parameters:
                image: The image to be rotated. (CV2 BGR Image)
                rotation_matrix: The matrix to rotate the image.
                output_img_box: The box that will correctly bound the rotated_image so the output image can fit inside it.
        """
        (aligned_x, aligned_y, aligned_w, aligned_h) = output_img_box
        return cv2.warpAffine(image, rotation_matrix, (aligned_w, aligned_h), flags=cv2.INTER_CUBIC)


    def align_and_extract_face(self, image, landmarks, output_size=(128,128), desired_left_eye_relative_position=(0.55, 0.55)):
        """
            Returns a face that's been aligned on the z-axis. The face is defined by the coordinates in the landmarks.

            Parameters:
                image: The image containing the face. (CV2 BGR Image)
                landmarks: An individual faces landmarks returns from get_faces_landmarks
                output_size: How big you want the output image to be (int(w), int(h))
                desired_left_eye_relative_position: Where you want the left eye to be positioned relative to the output image.
                                                    This should be a relative position to the output_size ( float(x), float(y) )
                                                    I recommend use the defaults: (0.55, 0.55), so most of the face and hair can still be seen.
        """

        # get the key landmark features
        lp, eye_center, facebox = landmarks['left-eye-center-pos'], landmarks['eye-center-pos'], landmarks['face-box']

        # calculating the center box for face box, this will also be the center for the output_box
        face_box_center = calc_box_center(facebox)

        # calculate the output box using the box center
        output_box = (face_box_center[0] - output_size[0] / 2, face_box_center[1] - output_size[1]/2, output_size[0], output_size[1])

        # left eye's absolute position relative to facebox
        current_left_eye_absolute_x_position = lp[0] - facebox[0]

        # desired left eye's absolute position relative to output box
        desired_left_eye_absolute_x_position = output_box[2] * desired_left_eye_relative_position[0]

        # scale needed would be the ratio between desired and current
        scale = desired_left_eye_absolute_x_position / current_left_eye_absolute_x_position

        # calculate alignment angle
        angle = self.determine_rotation_angle(landmarks)

        # rotation origin will be the eye_center
        rotation_origin = eye_center

        # calculate the rotation matrix
        rotation_matrix = self.determine_rotation_matrix(rotation_origin, angle, scale)

        # calculate the new bounding box needed to bound/contain the rotated image
        aligned_box = self.determine_bounding_box_of_rotated_box(calc_img_box(image), rotation_matrix)
        (aligned_x, aligned_y, aligned_w, aligned_h) = aligned_box
        # update the translation to fit the rotated image to the bounding box
        rotation_matrix[0, 2] -= aligned_x
        rotation_matrix[1, 2] -= aligned_y

        # perform the transformation
        warped_image = cv2.warpAffine(image, rotation_matrix, (aligned_w, aligned_h), flags=cv2.INTER_CUBIC)

        # calculate new box positions
        new_face_box_center = calc_rotate_point_with_rotation_matrix(face_box_center, rotation_matrix)
        new_output_box = [
                        int(new_face_box_center[0] - output_size[0] / 2),
                        int(new_face_box_center[1] - output_size[1]/2),
                        output_size[0], output_size[1]
        ]

        # calculate new left eye position
        new_lp = calc_rotate_point_with_rotation_matrix(lp, rotation_matrix)

        # calculate how far we are from the ideal y position for the left eye
        desired_left_eye_absolute_y_position = new_output_box[1] + int(new_output_box[3] * desired_left_eye_relative_position[1])
        distance = new_lp[1] - desired_left_eye_absolute_y_position

        # shift the y position such that the output box is in the desired y position
        new_output_box[1] += distance

        # it's possible output boxes's position will extend beyond current image space, pad the current image space such that they'll be enough space for the output box

        # figure out the padding
        x_left_pad = abs(min(0, new_output_box[0]))
        x_right_pad = abs(min(0, warped_image.shape[1] - (new_output_box[0] + new_output_box[2])))
        y_top_pad = abs(min(0, new_output_box[1]))
        y_bottom_pad = abs(min(0, warped_image.shape[0] - (new_output_box[1] + new_output_box[3])))

        # some amount of padding exists, create an image with enough padding space
        if x_left_pad + x_right_pad + y_top_pad + y_bottom_pad != 0:
            # create a new image with the padded distance
            new_shape = (warped_image.shape[0] + y_top_pad + y_bottom_pad,  warped_image.shape[1] + x_left_pad + x_right_pad, 3)
            padded_image = np.zeros(new_shape)

            # copy contents of warped image into padded image
            padded_image[y_top_pad:y_top_pad + warped_image.shape[0], x_left_pad:x_left_pad + warped_image.shape[1]] = warped_image

            # set the warped image to the padded image
            warped_image = padded_image

            # now "tare" the coordinates of the output box so that it fits in side the padded image
            new_output_box[0] += x_left_pad
            new_output_box[1] += y_top_pad

        # crop the output box out of the warped image
        cropped_image = warped_image[new_output_box[1]:new_output_box[1]+new_output_box[3], new_output_box[0]:new_output_box[0]+new_output_box[2]]

        # return the cropped image
        return cropped_image

    def calc_y_cross_angle(self, face_landmarks):
        """
            The "y-cross" is the angle between the center of the eyes, nose, and the mouth.
            A face in portrait position should have the center of the eyes, nose, and mouth be a near straight line. Meaning this angle will be near 180 degrees.
            A face facing the side, this line would form an angle, and the more side facing it is, the more the angle.
            For practical purposes, if the y_cross_angle is less than 130, the face is turning quite heavily on the side.
            If the cross_angle is less than 100, it is almost entirely facing the side.
        """

        # helper functions
        def vec_dotproduct(v1, v2):
            return np.sum(np.multiply(v1, v2))

        def vec_magnitude(v):
            return np.sum(np.power(v, 2)) ** 0.5

        def angle_between_2_vectors(v1, v2):
            dotprod =  vec_dotproduct(v1, v2)
            magprod = vec_magnitude(v1) * vec_magnitude(v2)
            return np.arccos([dotprod/magprod])[0]

        # main function
        def calc_angle_between_eye_center_nose_mouth(eye_center_pos, nose_pos, mouth_pos):

            nose_to_eye_vec = np.subtract(nose_pos, eye_center_pos)
            nose_to_mouth_vec = np.subtract(nose_pos, mouth_pos)
            return angle_between_2_vectors(nose_to_eye_vec, nose_to_mouth_vec)

        # calculate the angle in radians
        angle = calc_angle_between_eye_center_nose_mouth(face_landmarks['eye-center-pos'], face_landmarks['nose-pos'], face_landmarks['mouth-center-pos'])

        # convert to degrees
        return np.rad2deg(angle)


# import os
# from pathlib import Path
# ppdir = Path(__file__).resolve().parents[1].__str__()
# os.sys.path.append(ppdir)

# from NVXS_FaceDetector.NVXS_FaceDetector import NVXS_FaceDetector
# from CFA_FaceLandmarkDetector.CFA_FaceLandmarkDetector import CFA_LandmarkDetector
# from RCNN_FaceDetector.RCNN_FaceDetector import RCNN_FaceDetector

# aligner = Aligner()
# # face_detector = NVXS_FaceDetector()
# face_detector = RCNN_FaceDetector()
# landmark_detector = CFA_LandmarkDetector()

# image = cv2.imread('tests/images/58275.jpg')
# faceboxes = face_detector.detect_faceboxes(image)
# for i, facebox in enumerate(faceboxes):
#     face_landmarks = landmark_detector.detect_landmarks(image, facebox)
#     face_image = aligner.align_and_extract_face(image, face_landmarks)
#     cv2.imwrite('preview{}.jpg'.format(str(i)), face_image)
#     print(i, face_landmarks, aligner.calc_y_cross_angle(face_landmarks))
