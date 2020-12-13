import numpy
import data_ops
import model_ops


(x_train, y_train), (x_test, y_test) = data_ops.load_mnist_data()

model = model_ops.load()
x_my_test, y_my_test = data_ops.load_my_data()
model.evaluate(x_my_test, y_my_test)
