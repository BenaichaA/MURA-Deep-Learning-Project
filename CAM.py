import sys, getopt
import tensorflow as tf
import numpy as np
import cv2
import preprocess, mura_pipeline
from Models import DenseNet169
import matplotlib.pyplot as plt

img_rows, img_cols = 512, 512

def show_CAM(image_path, model_path):
    image_path_tensor = tf.placeholder(tf.string, name="image_path_tensor")
    label = tf.placeholder(tf.float32, shape=(None), name="label")

    image_dataset = tf.data.Dataset.from_tensor_slices(([image_path], [0]))
    image_dataset = image_dataset.map(lambda image, label: preprocess.parse_file(image, label, img_rows, img_cols))
    image_dataset = image_dataset.batch(1)

    # ====== Load altered model & compute activation maps ========
    model, FCL, last_conv = DenseNet169.densenet169_CAM_model(img_rows=img_rows, img_cols=img_cols, color_type=3, num_classes=1, dropout_rate=0)
    model.load_weights(model_path, by_name=True)
    # extract the Fully connected Layer(FCL) weights
    weights = FCL.get_weights()[0]
    weights = weights.reshape(weights.shape[0])
    # feed image through the NN and get the output of the last convolutional layer
    feature_maps = last_conv.predict(image_dataset, steps=1)[0]


    with tf.Session() as sess:
        # process original image in the same way, to be overlapped with the CAM
        image, label = sess.run(preprocess.parse_file(image_path_tensor, label, img_rows, img_cols), feed_dict={image_path_tensor: image_path, label:0})
        # compute the class activation map and resize to full image size
        class_activation_map = (feature_maps * weights).sum(axis=2)
        class_activation_map = cv2.resize(class_activation_map * 10, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        # display the image and class activation map
        plt.figure()
        plt.imshow(image.astype(int), cmap='gray')
        plt.imshow(class_activation_map, alpha=0.5)
        plt.grid(False)
        plt.show()

def main(argv):
    image_path = ''
    model_path = ''
    try:
        opts, args = getopt.getopt(argv,"i:m:")
    except getopt.GetoptError:
        print('CAM.py -i <path to image> -m <path to model weights>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            image_path = arg
        elif opt == "-m":
            model_path = arg

    show_CAM(image_path, model_path)

if __name__ == "__main__":
   main(sys.argv[1:])