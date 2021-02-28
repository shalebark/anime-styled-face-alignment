import _util as util
import cv2
from _face_detect import face_detect

# draw parts
def draw_facial_lines(img, face):
    """
        Returns a cv2 image with markers draw on key points.
        
        Parameters:
            img: A cv2 image
            face: An animeface face object
    """

    def draw_pos(p, img, marker=(255, 0, 0)): 
        x, y, w, h = p.x, p.y, p.width if hasattr(p, 'width') else 0, p.height if hasattr(p, 'height') else 0
        if w == 0 and h == 0:
            return cv2.circle(img, (x, y), 3, marker, 2)
        return cv2.rectangle(img, (x, y), (x + w, y + h), marker, 2)
    import numpy as np

    class pos:
        def __init__(self, x, y):
            self.x = x
            self.y = y    

    if hasattr(face, 'face'):
        img = draw_pos(face.face.pos, img, (125, 125, 0))
    if hasattr(face, 'left_eye'):
        img = draw_pos(face.left_eye.pos, img)
    if hasattr(face, 'right_eye'):
        img = draw_pos(face.right_eye.pos, img)
    if hasattr(face, 'mouth'):
        img = draw_pos(face.mouth.pos, img)
    if hasattr(face, 'nose'):
        img = draw_pos(face.nose.pos, img, (255, 0, 120))
    if hasattr(face, 'chin'):
        adj = face.chin.pos
        img = draw_pos(adj, img, (255, 120, 0))

    centers = util.calc_parts_center(face)

    # triangle for eyes and nose
    img = cv2.line(img, centers['left-eye-center'], centers['right-eye-center'], (0,0,255), 1)
    img = cv2.line(img, centers['left-eye-center'], centers['nose-center'], (0,0,255), 1)
    img = cv2.line(img, centers['right-eye-center'], centers['nose-center'], (0,0,255), 1)

    # triangle for center of eyes, top of the face bounding box, and the nose
    img = cv2.line(img, centers['eye-center'], centers['top-center'], (0,255,0), 1)
    img = cv2.line(img, centers['eye-center'], centers['nose-center'], (0,255,0), 1)
    img = cv2.line(img, centers['top-center'], centers['nose-center'], (0,255,0), 1)

    # triangle for eyes and mouth
    img = cv2.line(img, centers['left-eye-center'], centers['right-eye-center'], (122,122,0), 1)
    img = cv2.line(img, centers['left-eye-center'], centers['mouth-center'], (122,122,0), 1)
    img = cv2.line(img, centers['right-eye-center'], centers['mouth-center'], (122,122,0), 1)

    return img

def visualize(filepath=None, image=None):
    """
        Visualize the facial alignment.
        Returns the visualized image as a cv2 image

        Parameters:
            filepath: The path of the facial image. Use either filepath or image.
            image: The facial image, expects cv2 image. Use either filepath or image.
    """
    assert filepath is None or image is None, "Cannot use both filepath and image as parameters. Use either filepath or image."
    assert filepath or image, "Missing filepath or image."

    if filepath:
        image = util.read_image(filepath)

    face = face_detect(image)
    assert face is not False, "No faces found."
    return draw_facial_lines(image, face)

if __name__ == '__main__':
    import sys
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Visualize Alignment for Facial Features of Animated Characters.\n This is a debug tool.')
    parser.add_argument('images', metavar='Images', nargs='+', help='Filepaths to images with facial features to be aligned.')
    parser.add_argument('--dest', metavar='Destination', dest='destdir', help='Output directory. (default: current working directory)')

    args = parser.parse_args()
    images = args.images
    destdir = args.destdir if args.destdir is not None else os.getcwd()
        
    for image in images:
        try:
            imagepath = image if os.path.isabs(image) else os.path.join(os.getcwd(), image)
            destination_path = os.path.join(destdir, os.path.basename(image))
            visualized_image = visualize(filepath=imagepath)
            util.write_image(visualized_image, destination_path)
            # Message to stdout, success
            print('"{}" visualized. Save to: "{}"'.format(image, destination_path))
        except AssertionError as err:
            print('Unable to align image ' + image, file=sys.stderr)
            sys.stderr.writelines(str(err), file=sys.stderr)
            