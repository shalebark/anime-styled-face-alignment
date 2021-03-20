import numpy as np

def bound_box(bounding_box, bounded_box):
    bx, by, bw, bh = bounding_box
    cx, cy, cw, ch = bounded_box

    ox = int(max(bx, cx))
    oy = int(max(by, cy))

    ow = int(min(bx + bw, cx + cw) - ox)
    oh = int(min(by + bh, cy + ch) - oy)

    return (ox, oy, ow, oh)

def calc_img_box(img):
    return (0, 0, img.shape[1], img.shape[0])

def bound_img_box(img, bounded_box):
    return bound_box(img_box(img), bounded_box)

def expand_box_by_ratio(box, ratio):
    (x,y,w,h) = box
    pw = w * ratio[0] / 2
    ph = h * ratio[1] / 2
    return (int(x - pw), int(y - ph), int(w + ph), int(h + ph))

def calc_box_center(box):
    (x,y,w,h) = box
    return (int(x + w/2), int(y + h/2))

def calc_midway_point(p1, p2):
    return tuple(np.average([p1, p2], axis=0).astype(int))

def points_distance(p1, p2):
    return np.sum(np.subtract(p1, p2) ** 2) ** 0.5

def most_distance_point(origin, points):
    distances = points_distance(origin, points)
    return points[np.argmax(distances)]

def furthest_distance(origin, points):
    distances = points_distance(origin, points)
    return np.max(distances)

def angle_between_points(p1, p2):
    # figure out the adjacent
    x = [p1[0] - p2[0]]
    # figure out the opposite
    y = [p1[1] - p2[1]]
    # use arctan2 to get the angle
    return np.arctan2(y, x)[0]

def calc_rotate_point_with_rotation_matrix(point, rotation_matrix, scale=1):
    c = np.array(rotation_matrix).dot(np.array((point[0], point[1], 1)).reshape((3,1))).reshape(-1) * scale
    return tuple(c.astype(int))

def box_points(box):
    x,y,w,h = box
    p1 = (x, y) # top left
    p2 = (x + w, y) # top right
    p3 = (x, y + h) # bottom left
    p4 = (x + w, y + h) # bottom right
    return (p1, p2, p3, p4)
