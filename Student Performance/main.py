import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv('student-mat.csv', sep=";")

data = data[["G1", "G2", "G3", "age", "health","studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

best = 0
passedacc = []
iteration = 30

for x in range(iteration):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

    passedacc.append(accuracy)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

p = "G1"
q = "Trials"
style.use("ggplot")
pyplot.scatter(list(range(0, iteration)), passedacc)
pyplot.xlabel(q)
pyplot.ylabel("Accuracy")
pyplot.show()
