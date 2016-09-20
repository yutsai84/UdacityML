import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(reg.predict(X_test),y_test))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(reg2.predict(X_test),y_test))

results = {
 "Linear Regression": mae(reg2.predict(X_test),y_test),
 "Decision Tree": mae(reg.predict(X_test),y_test)
}
