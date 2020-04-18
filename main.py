import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import datasets

boston = datasets.load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=["MEDV"])

X = df
y = target

lm = linear_model.LinearRegression()
model = lm.fit(X, y)

#R squared
print(lm.score(X,y))

print(lm.coef_)

print(lm.intercept_)

predictions = lm.predict(X)

print(predictions[:10])