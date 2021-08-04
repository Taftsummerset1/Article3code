"""
Authors:
 - Nathan
Date: 15 2021 July
Aim:
Quickly checking a JSON if the event distribution change or spike around physical events.
"""
# linear regression feature importance
from sklearn.datasets import make_regression
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from matplotlib import pyplot
NAfile = "C:/Users/Work/Documents/PhD/Part 3/networkattributes.csv"
col_names = ['textlength', 'normalisedfavourites', 'normailisedimages', 'normalisedmentions', 'normalisedlinks', 'normalisedHashtags', 'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
          'userValue', 'tweetValue', 'lists', 'statuses', 'value']
# load dataset
pima = pd.read_csv("C:/Users/Work/Documents/PhD/Part 3/dump4.csv", header=None, names=col_names)
pima.head()
# define the columbs to be used as indepedant variables
feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks', 'normalisedHashtags', 'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
          'userValue', 'tweetValue', 'lists', 'statuses', 'value']
# define the features from the columbs or indepedant variables
X = pima[feature_cols]
# define the taget variable
y = pima.normalisedfavourites
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# define the model
model = LinearRegression()
# fit the model
model.fit(X_train, y_train)
# make prediction based on model
y_pred = model.predict(X_test)

# get importance
importance = model.coef_
# summarize feature importance
networkattributes = {}
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    networkattributes['Feature: %0d' %(i)] = v

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
print(networkattributes)
pyplot.show()
with open(NAfile, 'w') as f:
    for key in networkattributes.keys():
        f.write("%s,%s\n"%(key,networkattributes[key]))



