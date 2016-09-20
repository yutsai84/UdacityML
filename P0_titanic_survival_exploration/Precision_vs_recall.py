# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.metrics import confusion_matrix

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print "Decision Tree confusion matrix:",confusion_matrix(clf.predict(X_test),y_test)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(clf.predict(X_test),y_test),precision(clf.predict(X_test),y_test))


clf2 = GaussianNB()
clf2.fit(X_train, y_train)
print "GaussianNB confusion matrix:",confusion_matrix(clf2.predict(X_test),y_test)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(clf2.predict(X_test),y_test),precision(clf2.predict(X_test),y_test))

results = {
  "Naive Bayes Recall": recall(clf2.predict(X_test),y_test),
  "Naive Bayes Precision": precision(clf2.predict(X_test),y_test),
  "Decision Tree Recall": recall(clf.predict(X_test),y_test),
  "Decision Tree Precision": precision(clf.predict(X_test),y_test)
}
