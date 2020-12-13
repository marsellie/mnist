import tensorflow as tf
import cv2
import numpy
import os
from pathlib import Path

if __name__ == '__main__':
    print("ERR")


def load_my_data():
    data_folder = str(Path().absolute()) + "\\mydata"
    categories = dict({})

    for folder in os.listdir(data_folder):
        images = []
        for file in os.listdir(data_folder + "/" + folder):
            img = cv2.cvtColor(cv2.imread(data_folder + '/' + folder + '/' + file), cv2.COLOR_BGR2GRAY) / 255
            valid_image = numpy.empty(shape=(28, 28, 1))
            for i in range(0, len(img)):
                for j in range(0, len(img[i])):
                    valid_image[i][j] = numpy.array(img[i][j])
            images.append(valid_image)

        categories[folder] = numpy.array(images)
    return categories


def load_mnist_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
