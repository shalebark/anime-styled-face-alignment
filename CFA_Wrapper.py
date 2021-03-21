# # Credits - kanosawa
# # This is a slightly modified file of the Anime Face Landmark Detection file by Deep Cascaded Regression. https://github.com/kanosawa/anime_face_landmark_detection
# # Original File: https://github.com/kanosawa/anime_face_landmark_detection/blob/master/CFA.py

# 24 different landmarks
"""
* left/right from the character's perspective

0 : face-right-pos
1 : chin-pos
2 : face-left-pos
3 : right-brow-right-pos
4 : right-brow-middle-pos
5 : right-brow-left-pos
6 : left-brow-right-pos
7 : left-brow-middle-pos
8 : left-brow-left-pos
9 : nose-pos
10 : right-eye-top-right-pos
11 : right-eye-top-middle-pos
12 : right-eye-top-left-pos
13 : right-eye-bottom-pos
14 : right-eye-center-pos
15 : left-eye-top-right-pos
16 : left-eye-top-middle-pos
17 : left-eye-top-left-pos
18 : left-eye-bottom-pos
19 : left-eye-center-pos
20 : mouth-top-right-pos
21 : mouth-top-middle-pos
22 : mouth-top-left-pos
23 : mouth-bottom-pos

"""

# from _face_detect import face_detect

from NVXS_Wrapper import NVXS_Wrapper as NVXS
import _util as util
import cv2
import torch
from torchvision import transforms
import numpy as np

# git clone https://github.com/kanosawa/anime_face_landmark_detection
# checkpoint -- checkpoint_landmark_191116.pth.tar
from CFA import CFA

class CFA_Wrapper:
    def __init__(self, checkpoint='checkpoint_landmark_191116.pth.tar'):
        # params
        self._num_landmarks = 24
        self._landmark_detector = CFA(output_channel_num=self._num_landmarks + 1, checkpoint_name=checkpoint).cuda()
        self._nvxs = NVXS()

        # transform
        normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        train_transform = [transforms.ToTensor(), normalize]
        self._transformer = transforms.Compose(train_transform)

    def get_landmarks(self, image, faces_landmarks):
        """
            Returns CFA Landmarks.
            img: CV2 BGR Image
        """

        # convert to rgb image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = []
        for face in faces_landmarks:
            x_, y_, w_, h_ = face['face-box']

            # expand the face selection of the image (horizontally by 1/8 of original on both sides), (vertically, 1/4 of original going up)
            x = int(max(x_- w_ / 8, 0))
            rx = min(x_ + w_ * 9 / 8, img.shape[1])
            y = int(max(y_ - h_ / 4, 0))
            by = y_ + h_
            w = int(rx - x)
            h = int(by - y)

            # set image width (this should be the same size as the expected input data, which is 128x128x3)
            img_width = 128

            # crop and resize image
            cropped_img = img[y:y+h, x:x+w]
            facial_img = cv2.resize(cropped_img, (img_width, img_width), interpolation = cv2.INTER_CUBIC)

            # normalize and convert to tensors
            process_img = self._transformer(facial_img)
            process_img = process_img.unsqueeze(0).cuda()

            # get landmark classification heatmap
            heatmaps = self._landmark_detector(process_img)
            heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

            cfa_landmarks = []
            # calculate landmark position
            for i in range(self._num_landmarks):
                heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
                landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
                landmark_y = int(landmark[0] * h / img_width)
                landmark_x = int(landmark[1] * w / img_width)

                landmark = (x + landmark_x, y + landmark_y)
                cfa_landmarks.append(landmark)

            all_landmarks = face.copy()

            named_cfa_landmarks = {
                'face-right-pos': cfa_landmarks[0],
                'chin-pos': cfa_landmarks[1],
                'face-left-pos': cfa_landmarks[2],
                'right-brow-right-pos': cfa_landmarks[3],
                'right-brow-middle-pos': cfa_landmarks[4],
                'right-brow-left-pos': cfa_landmarks[5],
                'left-brow-right-pos': cfa_landmarks[6],
                'left-brow-middle-pos': cfa_landmarks[7],
                'left-brow-left-pos': cfa_landmarks[8],
                'nose-pos': cfa_landmarks[9],
                'right-eye-top-right-pos': cfa_landmarks[10],
                'right-eye-top-middle-pos': cfa_landmarks[11],
                'right-eye-top-left-pos': cfa_landmarks[12],
                'right-eye-bottom-pos': cfa_landmarks[13],
                'right-eye-center-pos': cfa_landmarks[14],
                'left-eye-top-right-pos': cfa_landmarks[15],
                'left-eye-top-middle-pos': cfa_landmarks[16],
                'left-eye-top-left-pos': cfa_landmarks[17],
                'left-eye-bottom-pos': cfa_landmarks[18],
                'left-eye-center-pos': cfa_landmarks[19],
                'mouth-top-right-pos': cfa_landmarks[20],
                'mouth-top-middle-pos': cfa_landmarks[21],
                'mouth-top-left-pos': cfa_landmarks[22],
            }

            all_landmarks.update(named_cfa_landmarks)

            landmarks.append(all_landmarks)

        return landmarks

    def get_faces_landmarks(self, image):
        # detect face
        faces_landmarks = self._nvxs.get_faces_landmarks(image)
        return self.get_landmarks(image, faces_landmarks)

    def visualize_landmarks(self, image):
        _, _2, landmarks = self.get_landmarks(image)
        output_img = image.copy()
        for i, landmark in enumerate(landmarks):
            output_img = cv2.putText(output_img, str(i), (landmark[0] - 2, landmark[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        return output_img

"""
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
have a relative distance you want between the eyes.
scale based on that distance.
we'll use some origin to scale, and that will serve as the "centering" of the image.
"""

# img = cv2.imread('/mnt/d/Workspace/non-non-biyori/131.jpg')
# cfa = CFA_Wrapper()
# landmarks, _, _2 = cfa.get_landmarks(img)

# output_img = align_image(img, landmarks)
# cv2.imwrite('output4.jpg', output_img)

# cv2.imwrite('output3.jpg', cfa.visualize_landmarks(img))
# print(face)
# x = tuple(face.skin.color.__dict__.values())
# print(  x )


# cfa = CFA_Wrapper()
# cv2.imwrite('output3.jpg', cfa.visualize_landmarks(img))
# exit()

# img = cv2.imread('/mnt/d/Workspace/non-non-biyori/16_rot.jpg')
# face = face_detect(img)
# print(face)
