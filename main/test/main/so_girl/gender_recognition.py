#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.
from dircache import listdir
import gzip

import os

import sys
from os.path import isfile, join
import cPickle
import zipfile

sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
from progressbar import ProgressBar
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image

import logging
import matplotlib.cm as cm
import pandas as pd

FACE_TRAINING_IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/faces")

FACE_TRAINING_LABEL_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "adiencedb/labels")


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

        labels = zip(file_name_list, gender_list, age_list)
        all_labels += labels

    df = pd.DataFrame(all_labels)
    writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
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
    labels = read_labels()

    for dirname, dirnames, filenames in os.walk(FACE_TRAINING_IMAGE_FOLDER):
        dirList = dirnames[:20]
        pbar = ProgressBar(maxval=len(dirList) or None).start()
        for subdirname in dirList:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if os.path.getsize(os.path.join(subject_path, filename)) > 80000:

                        try:
                            im = Image.open(os.path.join(subject_path, filename))
                            filename_for_label = filename.split(".", 2)[2]
                            label = [item for item in labels if item[0] == filename_for_label][0][1]
                            im = im.convert("L")
                            # resize to given size (if given)
                            im = im.resize((256, 256), Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(label)
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


def save(object, filename, protocol=-1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    zf = zipfile.ZipFile(filename + '.zip', 'w', zipfile.ZIP_DEFLATED)
    zf.writestr(filename + '.pkl', cPickle.dumps(object, -1))


def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()

    return object


if __name__ == "__main__":

    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    # if len(sys.argv) < 2:
    #     print "USAGE: facerec_demo.py </path/to/images>"
    #     sys.exit()
    # Now read in the image data. This must be a valid path!


    [X, y] = read_images()


    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=5)
    # Define the model as the combination
    my_model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):
    my_model.compute(X, y)
    # We then save the model, which uses Pythons pickle module:
    save_model('model.pkl', my_model)
    model = load_model('model.pkl')
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
        E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))
    # Plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet,
            filename="fisherfaces.png")
    # Perform a 10-fold cross validation
    cv = KFoldCrossValidation(model, k=10)
    cv.validate(X, y)
    # And print the result:
    cv.print_results()
