from dircache import listdir

from os.path import isfile, join
import os

__author__ = 'ramon'

import cv2


# use this script to install openCV https://github.com/jayrambhia/Install-OpenCV

PIC_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_pics")

FACE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_faces")

FACE_DETECT_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "facedetect")

profile_pics = [f for f in listdir(PIC_FOLDER) if isfile(join(PIC_FOLDER, f))]

face_cascade = cv2.CascadeClassifier(os.path.join(FACE_DETECT_FOLDER, 'haarcascade_frontalface_default.xml'))

for profile_pic in profile_pics:
    img = cv2.imread(os.path.join(PIC_FOLDER, profile_pic))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(FACE_FOLDER, profile_pic), roi_color)
