import numpy
import data_ops
import model_ops

# (x_train, y_train), (x_test, y_test) = data_ops.load_mnist_data()

my_data = data_ops.load_my_data()
model = model_ops.load()

for key in my_data.keys():
    results = model.predict(my_data[key])

    correct = 0
    for result in results:
        if key == str(numpy.argmax(result)):
            correct += 1
    print('for', key, 'valid', correct, '/', len(results), ';', 'acc:', float(correct) / len(results))
