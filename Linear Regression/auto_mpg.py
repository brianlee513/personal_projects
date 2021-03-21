import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("auto-mpg.data", sep = "\t")
data = data[["mpg", "cylinder", "displacement", "hp", "weight", "accel", "model_year", "origin"]]
print(data.head())

