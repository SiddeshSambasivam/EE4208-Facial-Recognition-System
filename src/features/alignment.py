import logging
import cv2 
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)

    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a

def rotate(img, nose_center, angle):
    M = cv2.getRotationMatrix2D(nose_center, angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return rotated

def rotation_detection(img, paths):

    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    fase_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_rects = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    face_rects = fase_cascade.detectMultiScale(gray, 1.3, 5)
    length_eyes = len(eyes_rects)

    if length_eyes == 2 and len(nose_rects) != 0 and len(face_rects) != 0:
        nose, right_eye, left_eye = get_eyes_nose(eyes_rects, nose_rects)
    else:
        logger.warning("Couldn't determine eyes/nose")
        return img

    center_of_forehead = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))
    center_pred = (int((face_rects[0][0] + face_rects[0][2]) / 2), int((face_rects[0][1] + face_rects[0][1]) / 2))
    length_line1 = distance(center_of_forehead, nose)
    length_line2 = distance(center_pred, nose)
    length_line3 = distance(center_pred, center_of_forehead)
    cos_a = cosine_formula(length_line1, length_line2, length_line3)
    angle = np.arccos(cos_a)
    rotated_point = rotate_point(nose, center_of_forehead, angle)
    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

    if is_between(nose, center_of_forehead, center_pred, rotated_point):
        angle = np.degrees(-angle)
    else:
        angle = np.degrees(angle)
    
    img = rotate(img, nose, angle)

    return img

