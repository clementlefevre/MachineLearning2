#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.
from dircache import listdir
import os
import sys
from os.path import isfile, join

from sklearn import svm, metrics

import cv2

sys.path.append("../..")
# import facerec modules
# import numpy, matplotlib and logging
from sklearn.cross_validation import train_test_split
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image

import pandas as pd

FACE_TRAINING_IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/faces")

FACE_TRAINING_LABEL_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/labels")


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_labels():
    labels_files = [f for f in listdir(FACE_TRAINING_LABEL_FOLDER) if isfile(join(FACE_TRAINING_LABEL_FOLDER, f))]
    all_labels = []
    for labels_file in labels_files:
        df = pd.read_csv(os.path.join(FACE_TRAINING_LABEL_FOLDER, labels_file), delimiter='\t', index_col=False,
                         header=0)

        # remove age <20 years

        df['age'] = df['age'].str.split(",").str[0]
        df['age'] = df['age'].str.lstrip('(')
        df.convert_objects(convert_numeric=True)
        df = df[df.age > 18]

        # remove undefined gender
        df = df[df.gender != 'u']

        # apply boolean is male
        df['gender'] = (df['gender'] == "m").astype(int)

        file_name_list = df.original_image.tolist()
        gender_list = df.gender.tolist()
        age_list = df.age.tolist()
        face_id_list = df.face_id.tolist()

        labels = zip(file_name_list, gender_list, age_list, face_id_list)
        all_labels += labels

    df = pd.DataFrame(all_labels)
    df.to_csv("labels_adiencedb.csv")
    return all_labels


def read_images(sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """

    X, y = [], []
    df_labels = pd.read_csv('labels_adiencedb.csv')
    labels = df_labels.values.tolist()

    for dirname, dirnames, filenames in os.walk(FACE_TRAINING_IMAGE_FOLDER):
        dirList = dirnames

        for subdirname in dirList:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if os.path.getsize(os.path.join(subject_path, filename)) > 80000:

                        try:
                            im = Image.open(os.path.join(subject_path, filename))
                            filename_for_label = filename.split(".", 2)[2]
                            person = [item for item in labels if item[1] == filename_for_label][0][4]
                            im = im.convert("L")
                            # resize to given size (if given)
                            im = im.resize((256, 256), Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(person)
                        except Exception:
                            pass
                # print "Could not find label for img :[" + filename_for_label + "]"

                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    # save(X, subdirname + "X.p")
    # save(y, subdirname + "y.p")
    print "found faces : {0} ".format(len(X))
    return [X, y]


if __name__ == "__main__":
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = "/test"
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:

    # Now read in the image data. This must be a valid path!
    [X, y] = read_images()

    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    model = cv2.createFisherFaceRecognizer()
    # Read
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model.train(np.asarray(X_train), np.asarray(y_train))
    # We now get a prediction from the model! In reality you
    # should always use unseen images for testing your model.
    # But so many people were confused, when I sliced an image
    # off in the C++ version, so I am just using an image we
    # have trained with.
    #
    # model.predict is going to return the predicted label and
    # the associated confidence:
    false = 0
    for i in range(len(X_test) - 1):

        [p_label, p_confidence] = model.predict(np.asarray(X_test[i]))

        # Print it:
        if p_label != y_test[i]:
            false += 1
    print "test size :%d" % (len(X_test))
    print "false :%d" % (false)


    # Cool! Finally we'll plot the Eigenfaces, because that's
    # what most people read in the papers are keen to see.
    #
    # Just like in C++ you have access to all model internal
    # data, because the cv::FaceRecognizer is a cv::Algorithm.
    #
    # You can see the available parameters with getParams():
    print model.getParams()
    # Now let's get some data:
    mean = model.getMat("mean")
    eigenvectors = model.getMat("eigenvectors")
    cv2.imwrite("test.png", X[0])
    # We'll save the mean, by first normalizing it:
    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(X[0].shape)
    # if out_dir is None:
    #     cv2.imshow("mean", mean_resized)
    # else:
    cv2.imwrite("mean.png", mean_resized)
    # Turn the first (at most) 16 eigenvectors into grayscale
    # images. You could also use cv::normalize here, but sticking
    # to NumPy is much easier for now.
    # Note: eigenvectors are stored by column:
    for i in xrange(len(np.unique(y)) - 1):
        eigenvector_i = eigenvectors[:, i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        eigenvector_i_colormap = cv2.applyColorMap(eigenvector_i_norm, cv2.COLORMAP_JET)
        # Show or save the images:


        cv2.imwrite("fisherface_%d.png" % i, eigenvector_i_colormap)


# with SVM

def run_gender_classifier():
    Xm, Ym = mkdataset('gender/male', 1)  # mkdataset just preprocesses images,
    Xf, Yf = mkdataset('gender/female', 0)  # flattens them and stacks into a matrix
    X = np.vstack([Xm, Xf])
    Y = np.hstack([Ym, Yf])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.1,
                                                        random_state=100)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)
    print("Results:\n%s\n" % (
        metrics.classification_report(
            Y_test, model.predict(X_test))))


def preprocess(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (100, 100))
    return cv2.equalizeHist(im)


def mkdataset(path, label):
    images = (cv2.resize(cv2.imread(fname), (100, 100))
              for fname in list_images(path))
    images = (preprocess(im) for im in images)
    X = np.vstack([im.flatten() for im in images])
    Y = np.repeat(label, X.shape[0])
    return X, Y
