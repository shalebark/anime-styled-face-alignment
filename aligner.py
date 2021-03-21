from NVXS_Wrapper import NVXS_Wrapper as NVXS
from CFA_Wrapper import CFA_Wrapper as CFA
import cv2
import numpy as np
import warnings
from geometric_utils import *


class Aligner:

    def __init__(self, use_cfa=True):
        self.nvxs = NVXS()
        self.detector = self.nvxs

        if use_cfa:
            try:
                self.cfa = CFA()
                self.detector = self.cfa
            except:
                warnings.warn('Unable to enable CFA, using NVXS.')


    def get_faces_landmarks(self, image):
        """
            Return an array of facial landmarks from the image.
            If no faces are detected, an empty array is given.

            Parameters:
                image: An CV2 BGR Image
        """
        return self.detector.get_faces_landmarks(image)

    def preprocess_landmarks(self, landmarks):
        """
            Returns a process landmarks to ensure all required positions are available.

            Parameters:
                landmarks: An individual faces landmarks returns from get_faces_landmarks
        """

        # generate the center positions of the eyes
        landmarks.update({
            'left-eye-center-pos': calc_box_center(landmarks['left-eye-box']),
            'right-eye-center-pos': calc_box_center(landmarks['right-eye-box']),
        })

        # generates the center position between the eyes
        landmarks['eye-center-pos'] = calc_midway_point(landmarks['left-eye-center-pos'], landmarks['right-eye-center-pos'])
        return landmarks

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

    def rotate_image_to_alignment(self, img, rotation_matrix, output_img_box):
        """
            Returns the rotated image, and the image will be correctly bounded so that the entire rotated image fits inside the output image.

            Parameters:
                img: The image to be rotated. (CV2 BGR Image)
                rotation_matrix: The matrix to rotate the image.
                output_img_box: The box that will correctly bound the rotated_image so the output image can fit inside it.
        """
        (aligned_x, aligned_y, aligned_w, aligned_h) = output_img_box
        return cv2.warpAffine(img, rotation_matrix, (aligned_w, aligned_h), flags=cv2.INTER_CUBIC)


    def align_image_main(self, img, landmarks, output_size, desired_left_eye_relative_position):
        """
            Returns a face that's been aligned on the z-axis. The face is defined by the coordinates in the landmarks.

            Parameters:
                img: The image containing the face. (CV2 BGR Image)
                landmarks: An individual faces landmarks returns from get_faces_landmarks
                output_size: How big you want the output image to be (int(w), int(h))
                desired_left_eye_relative_position: Where you want the left eye to be positioned relative to the output image.
                                                    This should be a relative position to the output_size ( float(x), float(y) )
                                                    I recommend use the defaults: (0.58, 0.58), so most of the face and hair can still be seen.
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
        aligned_box = self.determine_bounding_box_of_rotated_box(calc_img_box(img), rotation_matrix)
        (aligned_x, aligned_y, aligned_w, aligned_h) = aligned_box
        # update the translation to fit the rotated image to the bounding box
        rotation_matrix[0, 2] -= aligned_x
        rotation_matrix[1, 2] -= aligned_y

        # perform the transformation
        warped_image = cv2.warpAffine(img, rotation_matrix, (aligned_w, aligned_h), flags=cv2.INTER_CUBIC)

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

    def align_and_extract_face(self, img, landmarks, output_size=(128, 128), desired_left_eye_relative_position=(0.58, 0.58)):
        """
            Returns a facial image that's aligned on the z-axis. Image will be a CV2 BGR Image.

            Parameters:
                img: The image containing the face. (CV2 BGR Image)
                landmarks: An individual faces landmarks returns from get_faces_landmarks
                output_size: How big you want the output image to be (int(w), int(h))
                desired_left_eye_relative_position: Where you want the left eye to be positioned relative to the output image.
                                                    This should be a relative position to the output_size ( float(x), float(y) )
                                                    I recommend use the defaults: (0.58, 0.58), so most of the face and hair can still be seen.
        """
        landmarks = self.preprocess_landmarks(landmarks)
        return self.align_image_main(img, landmarks, output_size, desired_left_eye_relative_position)

aligner = Aligner(use_cfa=True)
img = cv2.imread('tests/images/moetron.jpg')
faces = aligner.get_faces_landmarks(img)
# print(faces)
output_img = aligner.align_and_extract_face(img, faces[0])
cv2.imwrite('preview.jpg', output_img)
