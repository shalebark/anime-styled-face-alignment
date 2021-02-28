import numpy as np
import _util as util
from _face_detect import face_detect

# https://towardsdatascience.com/precise-face-alignment-with-opencv-dlib-e6c8acead262

def _calculate_face_z_alignment_angle(centers):
    """
        Calculate the angle, the face is rotated in, in respect to the z-axis.
        Returns the angle of misalignment, in radians.

        Parameters:
            A dictionary of centers of facial components.
    """
    x = [centers['left-eye-center'][0] - centers['right-eye-center'][0]]
    y = [centers['left-eye-center'][1] - centers['right-eye-center'][1]]
    return np.arctan2(y, x)[0]

def _align_face(img, face):
    """
        Aligns image so that the face in the image is aligned, in the z-axis.

        Parameters:
            img: PIL image
            face: An animeface face object
    """
    centers = util.calc_parts_center(face)
    angle = _calculate_face_z_alignment_angle(centers)
    return util.rotate_image(img, angle, centers['nose-center'])

def align_face(filepath=None, image=None, axis='z'):
    """
        Aligns the facial image.
        Returns the image as a cv2 image

        Parameters:
            filepath: The path of the facial image. Use either filepath or image.
            image: The facial image, expects cv2 image. Use either filepath or image.
    """
    assert filepath is None or image is None, "Cannot use both filepath and image as parameters. Use either filepath or image."
    assert filepath or image, "Missing filepath or image."

    if filepath:
        image = util.read_image(filepath)
    face = face_detect(image)
    return _align_face(image, face)

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Aligns Facial Features of Animated Characters')
    parser.add_argument('images', metavar='Images', nargs='+', help='Filepaths to images with facial features to be aligned.')
    parser.add_argument('--dest', metavar='Destination', dest='destdir', help='Output directory. (default: current working directory)')

    args = parser.parse_args()
    images = args.images
    destdir = args.destdir if args.destdir is not None else os.getcwd()
        
    for image in images:
        try:
            imagepath = image if os.path.isabs(image) else os.path.join(os.getcwd(), image)
            aligned_image = align_face(filepath=imagepath)
            destination_path = os.path.join(destdir, os.path.basename(image))
            util.write_image(aligned_image, destination_path)
            # Message to stdout, success
            print('"{}" aligned. Save to: "{}"'.format(image, destination_path))
        except AssertionError as err:
            print('Unable to align image ' + image, file=sys.stderr)
            print(str(err), file=sys.stderr)