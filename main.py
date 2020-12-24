import data_ops
import model_ops
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def conf_matrix(model, x_test, y_test):
    predictions = model.predict(x_test)
    y_predict = []
    for i in range(len(predictions)):
        y_predict.append(predictions[i].argmax())
    uniq = sorted(set(y_test))

    cm = confusion_matrix(y_true=y_test, y_pred=np.array(y_predict))
    classes = []
    for i in range(len(uniq)):
        classes.append(str(uniq[i]))
    plot_confusion_matrix(cm, uniq)


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


model = model_ops.load()
x_test, y_test = data_ops.load_my_data()
conf_matrix(model, x_test, y_test)
