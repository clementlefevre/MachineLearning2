#!/usr/bin/env python
# -*- coding: utf-8 -*-
# with SVM

from dircache import listdir
import logging
import os
from shutil import copyfile
import sys
from os.path import isfile, join
import pickle
from time import time

from sklearn import svm, metrics
import cv2
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

sys.path.append("../..")

from sklearn.cross_validation import train_test_split
import numpy as np

try:
    from PIL import Image
except ImportError:
    import Image

import pandas as pd

import matplotlib.pyplot as plt

FACE_TRAINING_IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/faces")

FACE_TRAINING_LABEL_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/labels")

FACE_TRAINING_FEMALE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/gender/female")

FACE_TRAINING_MALE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/gender/male")

FACE_SO_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_faces")

FACE_SO_GIRLS_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_girls_faces")


def read_labels():
    labels_files = [f for f in listdir(FACE_TRAINING_LABEL_FOLDER) if isfile(join(FACE_TRAINING_LABEL_FOLDER, f))]
    all_labels = []
    for labels_file in labels_files:
        df = pd.read_csv(os.path.join(FACE_TRAINING_LABEL_FOLDER, labels_file), delimiter='\t', index_col=False,
                         header=0)
        # remove age <18 years
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


def split_gender():
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

                            filename_for_label = filename.split(".", 2)[2]
                            gender = [item for item in labels if item[1] == filename_for_label][0][2]
                            age = [item for item in labels if item[1] == filename_for_label][0][3]
                            if age > 17:
                                if gender == 0:
                                    copyfile(os.path.join(subject_path, filename),
                                             os.path.join(FACE_TRAINING_FEMALE_FOLDER, filename))
                                else:
                                    copyfile(os.path.join(subject_path, filename),
                                             os.path.join(FACE_TRAINING_MALE_FOLDER, filename))
                        except Exception:
                            pass
                            # print "Could not find label for img :[" + filename_for_label + "]"

                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise


def read_images(sz=None):
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
                            gender = [item for item in labels if item[1] == filename_for_label][0][2]
                            im = im.convert("L")
                            # resize to given size (if given)
                            im = im.resize((256, 256), Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(gender)
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
    return X, y


def run_gender_classifier():
    Xm, Ym = mkdataset(FACE_TRAINING_MALE_FOLDER, 1)  # mkdataset just preprocesses images,
    Xf, Yf = mkdataset(FACE_TRAINING_FEMALE_FOLDER, 0)  # flattens them and stacks into a matrix
    X = np.vstack([Xm, Xf])
    Y = np.hstack([Ym, Yf])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.1,
                                                        random_state=100)
    model = svm.SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
                    decision_function_shape=None, degree=3, gamma=0.005, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
    model.fit(X_train, Y_train)
    print("Results:\n%s\n" % (
        metrics.classification_report(
            Y_test, model.predict(X_test))))

    print metrics.confusion_matrix(Y_test, model.predict(X_test))
    pickle.dump(model, open("gender_classifier.p", "wb"))


def extract_girl():
    # read all files in so_faces
    all_so_faces = list_images(FACE_SO_FOLDER)
    # load model
    model = pickle.load(open("gender_eigenfaces.p", "rb"))

    # if is girl then save in so_girls_faces
    for face in all_so_faces:
        image = cv2.imread(face)

        image = preprocess(image)
        X_test_pca = pca.transform(X_test)
        result = model.predict(image)
        if result == 0:
            print  "This guy {0} is a girl ".format(face)
            copyfile(face,
                     os.path.join(FACE_SO_GIRLS_FOLDER, face.split('/')[-1]))
        else:
            print  "This guy {0} is a boy ".format(face)


def preprocess(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im = cv2.resize(im, (256, 256))

    equalized = cv2.equalizeHist(im)

    return equalized


def list_images(path):
    all_images = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return all_images


def mkdataset(path, label):
    images = (cv2.resize(cv2.imread(fname), (100, 100))
              for fname in list_images(path))
    images = (preprocess(im) for im in images)
    X = np.vstack([im.flatten() for im in images])
    Y = np.repeat(label, X.shape[0])
    return X, Y


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


def eingenfaces():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    Xm, Ym = mkdataset(FACE_TRAINING_MALE_FOLDER, 1)  # mkdataset just preprocesses images,
    Xf, Yf = mkdataset(FACE_TRAINING_FEMALE_FOLDER, 0)  # flattens them and stacks into a matrix
    X = np.vstack([Xm, Xf])
    y = np.hstack([Ym, Yf])

    n_features = X.shape[1]
    h, w = 256, 256

    # the label to predict is the id of the person
    target_names = np.array(['female', 'male'])
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)


    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 256

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))


    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)


    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    pickle.dump(clf, open("gender_eigenfaces.p", "wb"))

    ###############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the result of the prediction on a portion of the test set

    def title(y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()

    # read all files in so_faces
    all_so_faces = list_images(FACE_SO_FOLDER)
    # load model
    model = pickle.load(open("gender_eigenfaces.p", "rb"))

    # if is girl then save in so_girls_faces
    for face in all_so_faces:
        image = cv2.resize(cv2.imread(face), (100, 100))

        img = preprocess(image)
        X = np.vstack([img.flatten()])
        X_pca = pca.transform(X)
        result = model.predict(X_pca)
        if result == 0:
            print  "This guy {0} is a girl ".format(face)
            copyfile(face,
                     os.path.join(FACE_SO_GIRLS_FOLDER, face.split('/')[-1]))
        else:
            print  "This guy {0} is a boy ".format(face)


if __name__ == "__main__":
    # run_gender_classifier()
    # split_gender()
    eingenfaces()
    # extract_girl()
