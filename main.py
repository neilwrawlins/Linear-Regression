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

#For 1 unit of change on X axis, y changes a percentage between 0% - 100%
print(lm.coef_)

#Y-Intercept
print(lm.intercept_)

#predict Y (dependent) values base off of X (independent) values
predictions = lm.predict(X)

#Print out first 10 predictions (0-9)
print(predictions[:10])
