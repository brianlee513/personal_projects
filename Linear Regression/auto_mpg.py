import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("auto-mpg.data", sep="\t")
data = data[["mpg", "cylinder", "displacement", "hp", "weight", "accel", "model_year", "origin"]]

data = data[data["hp"] != '?']
data["hp"] = (data["hp"]).astype('int64')


predict = "mpg"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0
passed_accuaracy = []
iteration = 30
for i in range(iteration):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("auto-mpg.pickle", "wb") as f:
            pickle.dump(linear, f)

    passed_accuaracy.append(acc)

pickle_in = open("auto-mpg.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])
    delta = abs(prediction[x] - y_test[x])
    print(delta)
p = "mpg"
q = "Trials"
style.use("ggplot")
pyplot.scatter(list(range(0, iteration)), passed_accuaracy)
pyplot.xlabel(q)
pyplot.ylabel("Accuracy")
pyplot.show()
