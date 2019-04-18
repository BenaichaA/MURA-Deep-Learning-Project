import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import sys, getopt
import preprocess, mura_pipeline
from Models import DenseNet169


img_rows, img_cols = 224, 224


def train(n_epochs, batch_size, starting_weights_path, save_path):
    train_images, train_labels, valid_images, valid_labels, train_set, valid_set = mura_pipeline.getDataSets(batch_size, img_rows, img_cols)

    model = DenseNet169.densenet169_model(img_rows=img_rows, img_cols=img_cols, color_type=3, num_classes=1, dropout_rate=0.2)

    model.load_weights(starting_weights_path, by_name=True)

    model.fit(train_set, epochs=n_epochs, steps_per_epoch=(len(train_images)//batch_size), validation_data=valid_set, validation_steps=len(valid_images)//batch_size)

    model.save(save_path)


def main(argv):
    starting_weights_path = "./Saved_Models/Dense169_ImageNet.h5"
    save_path = strftime("./Saved_Models/Dense169_%Y%m%d%H%M%S.h5", gmtime())
    n_epochs = 1
    batch_size = 8
    try:
        opts, args = getopt.getopt(argv,"n:b:s:f:")
    except getopt.GetoptError:
        print('train.py -n <number of epochs> -b <batch size> -s <path to starting weights> -f <save model path> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-n":
            n_epochs = int(arg)
        elif opt == "-b":
            batch_size = int(arg)
        elif opt == "-s":
            starting_weights_path = arg
        elif opt == "-f":
            save_path = arg
    train(n_epochs, batch_size, starting_weights_path, save_path)

if __name__ == "__main__":
   main(sys.argv[1:])