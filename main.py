import tensorflow as tf
import matplotlib as plt
import cv2


def train(x, y):
    trained_model = tf.keras.models.Sequential()
    trained_model.add(tf.keras.layers.Conv2D(28, input_shape=(28, 28, 1), kernel_size=(3, 3)))
    trained_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    trained_model.add(tf.keras.layers.Flatten())
    trained_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    trained_model.add(tf.keras.layers.Dropout(0.2))
    trained_model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    trained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    trained_model.fit(x, y, epochs=10)
    trained_model.save('model')
    return trained_model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img = cv2.imread("mydata/1.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_norm = img_gray/255
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# model = train(x_train, y_train)
model = tf.keras.models.load_model('model')
# val_loss, val_acc = model.evaluate(x_test, y_test)

print(model.predict(img_norm.reshape(img_norm.shape[0], 28, 28, 1)))
