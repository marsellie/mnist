import tensorflow as tf


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


def load():
    return tf.keras.models.load_model('model')
