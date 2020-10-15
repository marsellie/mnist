import tensorflow as tf
import matplotlib as plt
import cv2
import numpy


def train(x, y):
    trained_model = tf.keras.models.Sequential()
    trained_model.add(tf.keras.layers.Conv2D(28, input_shape=(28, 28, 1), kernel_size=(3, 3)))
    trained_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    trained_model.add(tf.keras.layers.Flatten())
    trained_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    trained_model.add(tf.keras.layers.Dropout(0.2))
    trained_model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    trained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    trained_model.fit(x, y, epochs=3)
    trained_model.save('model')
    return trained_model


def load_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


# (x_train, y_train), (x_test, y_test) = load_data()

img = cv2.cvtColor(cv2.imread("mydata/2.png"), cv2.COLOR_BGR2GRAY) / 255
valid_image = numpy.empty(shape=(28, 28, 1))

for i in range(0, len(img)):
    for j in range(0, len(img[i])):
        valid_image[i][j] = numpy.array(img[i][j])

my_data = numpy.array([valid_image])

# model = train(x_train, y_train)
model = tf.keras.models.load_model('model')
result = model.predict(my_data)
# val_loss, val_acc = model.evaluate(x_test, y_test)
print(numpy.argmax(result))
