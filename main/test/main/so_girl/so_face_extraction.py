import os

from progressbar import ProgressBar

__author__ = 'ramon'

import cv2


# use this script to install openCV https://github.com/jayrambhia/Install-OpenCV

PIC_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/aligned")

FACE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_faces")

FACE_DETECT_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "facedetect")

TARGET_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/faces")

face_cascade = cv2.CascadeClassifier(os.path.join(FACE_DETECT_FOLDER, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(FACE_DETECT_FOLDER, 'haarcascade_eye.xml'))


class Face_extractor():
    def main(self):
        list_pictures = self.load_pictures()
        self.detect_face(list_pictures)

    def load_pictures(self):
        list_pictures = []
        for dirname, dirnames, filenames in os.walk(PIC_FOLDER):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    list_pictures.append((subject_path, subdirname, filename))

        return list_pictures

    def detect_face(self, list_pictures):
        pbar = ProgressBar(maxval=len(list_pictures) or None).start()
        for profile_pic in list_pictures:
            img = cv2.imread(os.path.join(profile_pic[0], profile_pic[2]))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                # for (ex, ey, ew, eh) in eyes:
                #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                if len(eyes) > 0:
                    self.save_face(roi_color, profile_pic)

    def save_face(self, roi_color, profile_pic):
        target_folder = os.path.join(TARGET_FOLDER, profile_pic[1])
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        cv2.imwrite(os.path.join(target_folder, profile_pic[2]), roi_color)


if __name__ == '__main__':
    Face_extractor().main()
