import numpy as np
import tensorflow as tf
import preprocess
category = "XR_HAND"

def getDataSets(batch_size, row, col):
    # ============= Load Data ==================
    train_images = np.load('train_paths_array.npy')
    train_labels = np.load('train_labels_array.npy').astype(np.int32)

    valid_images = np.load('valid_paths_array.npy')
    valid_labels = np.load('valid_labels_array.npy').astype(np.int32)

    train_images, train_labels = preprocess.shuffle(train_images, train_labels)

    # ============= Filter by Category ==================

    # train_category_indices = np.core.defchararray.find(train_images, category)!=-1
    # valid_category_indices = np.core.defchararray.find(valid_images, category)!=-1

    # train_images = train_images[train_category_indices]
    # train_labels = train_labels[train_category_indices]
    # valid_images = valid_images[valid_category_indices]
    # valid_labels = valid_labels[valid_category_indices]

    # ============= Data Pipeline ==================
    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_set = train_set.map(lambda image, label: preprocess.parse_and_augment(image, label, row, col), num_parallel_calls=6)
    train_set = train_set.batch(batch_size=batch_size)
    train_set = train_set.prefetch(buffer_size=2*batch_size)
    train_set = train_set.repeat()


    valid_set = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    valid_set = valid_set.map(lambda image, label: preprocess.parse_file(image, label, row, col), num_parallel_calls=6)
    valid_set = valid_set.batch(batch_size=batch_size)
    valid_set = valid_set.prefetch(buffer_size=2*batch_size)
    valid_set = valid_set.repeat()

    return (train_images, train_labels, valid_images, valid_labels, train_set, valid_set)
