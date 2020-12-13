import numpy
import data_ops
import model_ops


model = model_ops.load()
x_test, y_test = data_ops.load_my_data()
model.evaluate(x_test, y_test)