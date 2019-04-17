import csv
import numpy as np
import tensorflow as tf

def format_labels():
    paths = "MURA-v1.1/valid_image_paths.csv"
    labels = "MURA-v1.1/valid_labeled_studies.csv"

    filePaths = []

    with open(paths) as pathsfile:
        readPaths = csv.reader(pathsfile, delimiter=',')
        for row in readPaths:
            filePaths.append(row[0])

    filePaths = np.array(filePaths)
    fileLabels = np.zeros(filePaths.size)


    with open(labels) as labelsfile:
        readLabels = csv.reader(labelsfile, delimiter=',')
        for row in readLabels:
            fileLabels[np.flatnonzero(np.core.defchararray.find(filePaths, row[0])!=-1)] = row[1]

    np.save("valid_labels_array.npy", fileLabels)
    np.save("valid_paths_array.npy", filePaths)

def format_studies():
    labels = "MURA-v1.1/valid_labeled_studies.csv"
    filePaths = []

    with open(labels) as pathsfile:
        readPaths = csv.reader(pathsfile, delimiter=',')
        for row in readPaths:
            filePaths.append([row[0], int(row[1])])

    np.save("valid_studies_array.npy", filePaths)


def parse_and_augment(filename, label, row, col):
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(tf.image.resize_image_with_pad(image, row, col), tf.float32)
    image = tf.contrib.image.rotate(image, angles=tf.random.uniform([1], minval=-0.3, maxval=0.3)[0], interpolation='BILINEAR')
    image = tf.image.random_flip_left_right(image)
    image = tf.image.grayscale_to_rgb(image)
    return image, label

def parse_file(filename, label, row, col):
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(tf.image.resize_image_with_pad(image, row, col), tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    return image, label

def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

