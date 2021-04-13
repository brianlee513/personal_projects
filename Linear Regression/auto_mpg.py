import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("auto-mpg.data", sep="\t")
data = data[["mpg", "cylinder", "displacement", "hp", "weight", "accel", "model_year", "origin"]]

data = data[data["hp"] != '?']
data["hp"] = (data["hp"]).astype('int64')


predict = "mpg"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)
print(acc)
