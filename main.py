import numpy
import data_ops
import model_ops


my_data = data_ops.load_my_data()
model = model_ops.load()
results = model.predict(my_data)

for result in results:
    print(numpy.argmax(result))
