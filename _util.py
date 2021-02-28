import PIL.Image as Image
import numpy as np
import cv2

def calc_parts_center(face):
    """
        Returns the centers of facial parts as a dictionary.
            {left-eye-center: (x,y), right-eye-center: (x,y), nose-center: (x,y), mouth-center: (x,y), chin-center: (x,y), eye-center: (x,y), top-center: (x,y)}

        Parameters:
            face: An animeface face object.
    """
    def calc_center(p):
        x, y, w, h = p.x, p.y, p.width if hasattr(p, 'width') else 0, p.height if hasattr(p, 'height') else 0
        return (int((x + x + w) / 2), int((y + y + h) / 2))

    lec = calc_center(face.left_eye.pos)
    rec = calc_center(face.right_eye.pos)
    nec = calc_center(face.nose.pos)
    mec = calc_center(face.mouth.pos)
    cec = calc_center(face.chin.pos)

    bec = (int((lec[0] + rec[0]) / 2), int((lec[1] + rec[1]) / 2)) # center of both eyes
    tcc = (int((face.face.pos.x + face.face.pos.x + face.face.pos.width) / 2), face.face.pos.y) # top center position
    return {
        'left-eye-center': lec, 'right-eye-center': rec, 'nose-center': nec, 'mouth-center': mec, 'chin-center': cec, 
        'eye-center': bec, 'top-center': tcc
    }

def calc_distance(a, b):
    """
        Returns the distance between 2 points.

        Parameters:
            a: Point 1 (x,y)
            b: Point 2 (x,y)
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def calc_rotated_position(origin, point, angle):
    """
        Calculates the position of parameter {point} after its been rotated around the {origin} by {angle} (in radians).
        Returns the rotated position of point. (x,y)

        Parameters:
            origin: Point of Rotation (x,y)
            point: Point to be rotated (x,y)
            angle: angle to be rotated, in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def convert_to_pil_to_cv2_image(pil_image):
    """
        Converts a PIL Image to a cv2 image. 
        Returns a cv2 image.
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def convert_to_cv2_to_pil_image(cv2_image):
    return Image.fromarray(np.uint8(cv2_image)).convert('RGB')


def rotate_image(image, angle, origin = None, axis='z'):
    """
        Rotates a cv2 image by angle (in radians), in a certain axis.
        Returns the rotated image as cv2 image.

        Parameters:
            image: A cv2 image to be rotated.
            angle: The angle of rotation in radians.
            origin: The point in which the image is rotated around. If None is provided, the center of the image is the origin.
            axis: Currently only the z-axis is supported.
    """
    (h, w) = image.shape[:2]

    if origin is None:
        origin = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(origin, angle * 180 / np.pi, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def read_image(filepath):
    """
        Returns the image from the filepath as cv2 image.

        Parameters:
            filepath: filepath to read from
    """
    return cv2.imread(filepath)

def write_image(img, savepath):
    """
        Writes the image to disk.
        Parameters:
            img: cv2 image. 
            savepath: filepath to save to.
    """
    cv2.imwrite(savepath, img)


# def is_inside_polygon(between_point, polygon_points):
#     """
#         Returns if between_point is inside a polygon.

#         Parameters:
#             between_point: The point that is verified if is within the list of points. 
#             polygon_points: An array of points, that are the vertexes of the polygon. The length of the array must be atleast 3.
#     """
#     assert len(polygon_points) >= 3, "polygon_points is too small, the length of the array must be atleast 3"

#     c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
#     c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
#     c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
#     if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
#         return True
#     else:
#         return False