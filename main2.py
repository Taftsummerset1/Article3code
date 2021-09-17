"""
"""
import json
import csv
from uuid import uuid4
from datetime import datetime
import time
from pprint import pprint

# linear regression feature importance
from sklearn.datasets import make_regression
import pandas as pd
import cmath as math
import sys
from matplotlib.ticker import MaxNLocator
from numpy import unique
from numpy import where
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

"""These inputs will change the output of the script, also needs to be tailored to the datasets being used

---IMPORTANT--- 

This paramaters must be set correctly for the dataset type and number to produce the correct graphs, plots and confusion matrix

Each value and dataset-type will change the functions and type of process used
"""
network_identifier = 'Nintendo'
dataset1type = 'TweetBinder' #this determines what reader preprocesses the data. Options are 'Tweetbinder', 'Eurovision4', 'Pakelect'
dataset2type = 'TweetBinder' #this determines what reader preprocesses the data. Options are 'Tweetbinder', 'Eurovision4', 'Pakelect'
dataset3type = 'TweetBinder' #this determines what reader preprocesses the data. Options are 'Tweetbinder', 'Eurovision4', 'Pakelect'
use_entirecampaign = False #if True, all other switches must be false, runs entire dataset and outputs 1 CNA
use_time = False # Only if above is False, if True, can either time divide or runs dataset via time block - Must change time delta as well.
use_rows = False # if true, runs dataset in smaller blocks sequentially and provide a CNA for each block, must change no_of_rows_per_run accordinly.
use_hashtags = True #if true, will pull out each hashtag and generate a CNA for each hashtag within dataset
use_2_datasets = True #if true, will run two datasets either via row, hashtag or time, and plot the CNAs on scatter plot with clustering
use_3_datasets = False #if true, will run three datasets - IMPORTANT - use_2_datasets must also be true. Only working for hashtag and Rows
scatter2D = False # if true, will plot the two CNAs against whatever subspace has been identified.
plot3d = False # if true, will plot in 3d after linear regression
classify = True #use the NN classifier to determine the types of campaigns
no_of_rows_per_run = 10 #used to divide the main campaign
no_of_tweets_per_hashtag = 50 #defines the minimum number of tweets that can generate a hashtag - data division becomes important for model training in LR / IC function
time_delta = 3600 #in seconds (unix , 3600 = 1 hour, 86400 = day)
K = 2 # number of campaigns being clustered
""" These parameters do not effect processing """
campaigntype1 = 'Sporting' #first dataset name / type of campaign (political, entertainment, culture, etc)
campaigntype2 = 'Political' #Second dataset type of campaign (political, entertainment, culture, etc)
campaigntype3 = 'Conflict2' #Third dataset type of campaign (political, entertainment, culture, etc)

""" These preset values will impact preprocessing only """
cumulativeevent = 0
cumulativeimg = 0
cumulativelink = 0
cumulativehash = 0
eventcounter = 0
lastdate = []
data = []

def preprocess_to_list_Nintendo_and_Eurovision(input_json):
    retdata = []
    for i in input_json:
        exculsion = len(i)
        if exculsion > 10:
            mentions = []
            hashtaglist = []
            url = []
            createdAt = i['created_at']

            urllist = i['entities']['urls']
            for u in urllist:
                url.append(u['url'])#

            retweeted = i['retweeted']
            if retweeted:
                retweetcount = i['retweet_count']
            else:
                retweetcount = 0

            favorited = i['favorited']
            if favorited:
                favoritecount = i['favorited_count']
            else:
                favoritecount = 0

            text = i['text']
            textlength = len(text)
            followers = i['user']['followers_count']
            user = i['user']['screen_name']
            following = i['user']['friends_count']
            hashtagrepo = i['entities']['hashtags']
            for h in hashtagrepo:
                hashtaglist.append(h['text'])
            replycount = i['reply_count']
            retweetcount = i['retweet_count']
            favoritecount = i['favorite_count']
            mentionlist = i['entities']['user_mentions']

            for m in mentionlist:
                mentions.append(m['name'])
            try:
                ftFration = (followers / following)
            except:
                ftFration = 0
            try:
                normalisedfavourites = (favoritecount / followers)
            except:
                normalisedfavourites = 0
            try:
                normalisedmentions = (len(mentions) / followers)
            except:
                normalisedmentions = 0
            try:
                normalisedlinks = (len(url) / followers)
            except:
                normalisedlinks = 0
            try:
                normalisedHashtags = (len(hashtaglist) / followers)
            except:
                normalisedHashtags = 0
            try:
                normalisedRetweets = (retweetcount / followers)
            except:
                normalisedRetweets = 0
            try:
                normalisedReplies = (replycount / followers)
            except:
                normalisedReplies = 0
            try:
                normalisedimages = (images / followers)
            except:
                normalisedimages = 0
            constructed_list = [textlength, normalisedfavourites, normalisedimages, normalisedmentions,
                                normalisedlinks,
                                normalisedHashtags, normalisedRetweets, normalisedReplies,  retweetcount,favoritecount, hashtaglist, createdAt]
            retdata.append(constructed_list)
            print(constructed_list)
        if retweetcount != 0:
            exit()
            #TODO must make the feature lists from all datasets consistent otherwise, this will skew the confusion / AI results.
    return retdata

def preprocess_to_list_Pakelect(input_json):
    retdata = []
    for i in input_json:
        exculsion = len(i)
        if exculsion > 10:
            mentions = []
            hashtaglist = []
            url = []
            createdAt = i['created_at']
            #images = i['images']
            urllist = i['entities']['urls']
            for u in urllist:
                url.append(u['url'])
            retweeted = i['retweeted']
            text = i['text']
            textlength = len(text)
            followers = i['user']['followers_count']
            user = i['user']['screen_name']
            following = i['user']['friends_count']
            hashtagrepo = i['entities']['hashtags']
            for h in hashtagrepo:
                hashtaglist.append(h['text'])
            retweetcount = i['retweet_count']
            favorites = i['favorite_count']
            mentionlist =  i['entities']['user_mentions']
            try:
                statuses = i['statuses_count']
            except:
                statuses = 0
            try:
                lists = i['listed_count']
            except:
                lists = 0
            for m in mentionlist:
                mentions.append(m['name'])
            try:
                ftFration = (followers / following)
            except:
                ftFration = 0
            try:
                normalisedfavourites = (favorites / followers)
            except:
                normalisedfavourites = 0
            try:
                normalisedmentions = (len(mentions) / followers)
            except:
                normalisedmentions = 0
            try:
                normalisedlinks = (len(url) / followers)
            except:
                normalisedlinks = 0
            try:
                normalisedHashtags = (len(hashtaglist) / followers)
            except:
                normalisedHashtags = 0
            try:
                normalisedRetweets = (retweetcount / followers)
            except:
                normalisedRetweets = 0
            try:
                normalisedReplies = (replycount / followers)
            except:
                normalisedReplies = 0
            normalisedimages = 0
            sentiment = 0
            originals = 0
            publicationscore = 0
            userValue = 0
            tweetValue = 0
            value = 0
            actualhashtags = 0

            constructed_list = [textlength, normalisedfavourites, normalisedimages, normalisedmentions,
                                normalisedlinks,
                                normalisedHashtags, normalisedRetweets, normalisedReplies, sentiment, originals,
                                publicationscore,
                                userValue, tweetValue, lists, statuses, value, hashtaglist, createdAt]
            retdata.append(constructed_list)
            #TODO must make the feature lists from all datasets consistent otherwise, this will skew the confusion / AI results.
    return retdata

def preprocess_to_list_nonTweetBinder(input_json):
    retdata = []
    limit = []
    eventcounter = 0
    for i in input_json:
        exculsion = len(i)
        if exculsion > 1:
            eventcounter = eventcounter + 1
            mentions = []
            hashtaglist = []
            urllist = []
            createdAt = i['created_at']
            #url = i['user']['url'] #only needed if using the URL links for analysis
            # print(url)
            retweeted = i['retweetedTweet']
            # print(retweeted)
            text = i['text']
            # print(text)
            textlength = len(text)
            # print(textlength)
            followers = i['user']['followers_count']
            # print(followers)
            user = i['user']['screen_name']
            # print(user)
            following = i['user']['friendsCount']
            # print(following)
            #hashtagrepo = i['entities']['hashtags']
            #for h in hashtagrepo:
                #hashtaglist.append(h['text'])
            # print(hashtaglist)
            replycount = i['replyCount']
            # print(replycount)
            retweetcount = i['retweetCount']
            # print(retweetcount)
            favorites = i['likeCount']
            try:
                mentionlist = i['mentionedUsers']
            except:
                mentionlist = []
            # print(mentionlist)
            try:
                ftFration = (followers / following)
            except:
                ftFration = 0
            try:
                normalisedfavourites = (favoritecount / followers)
            except:
                normalisedfavourites = 0

            try:
                normalisedmentions = (mentions / followers)
            except:
                normalisedmentions = 0
            try:
                normalisedlinks = (len(urllist) / followers)
            except:
                normalisedlinks = 0
            try:
                normalisedHashtags = (len(hashtags) / followers)
            except:
                normalisedHashtags = 0
            try:
                normalisedRetweets = (retweeted / followers)
            except:
                normalisedRetweets = 0
            normalisedimages = 0
            normalisedReplies = 0
            sentiment = 0
            originals = 0
            publicationscore = 0
            userValue = 0
            tweetValue = 0
            lists = 0
            statuses = 0
            value = 0
            actualhashtags = 0

            constructed_list = [textlength, normalisedfavourites, normalisedimages, normalisedmentions, normalisedlinks,
                                normalisedHashtags, normalisedRetweets, normalisedReplies, sentiment, originals,
                                publicationscore,
                                userValue, tweetValue, lists, statuses, value, hashtaglist, createdAt]
            # print(constructed_list)
            retdata.append(constructed_list)
        # print('this is retdata', retdata)
    return retdata
def preprocess_to_list_tweetbinderonly(input_json):
    """
    Authors:
     - Nathan
    Date: 15 2021 July
    Aim:
    Using the Tweetbinder JSON datastructure, read in the JSON and pull out features of each tweet.
    """
    retdata = []

    for i in input_json:
        if len(i) > 10:  #ignores small tweets at end of json
            constructed_list = []
            count = 0
            userID = i['_id']
            createdAt = i['createdAt']
            textlength = i['counts']['textLength']
            sentiment = i['counts']['sentiment']
            favorites = i['counts']['favorites']
            images = i['counts']['images']
            mentions = i['counts']['mentions']
            links = i['counts']['links']
            hashtags = i['counts']['hashtags']
            retweets = i['counts']['retweets']
            totalretweets = i['counts']['totalRetweets']
            originals = i['counts']['originals']
            replies = i['counts']['replies']
            publicationscore = i['counts']['publicationScore']
            userValue = i['counts']['userValue']
            tweetValue = i['counts']['tweetValue']
            countfollowers = i['user']['followers']
            countfollowing = i['user']['following']
            lists = i['user']['counts']['lists']
            value = i['user']['value']
            actualhashtags = i['hashtags']
            createdAtdaydate = datetime.fromtimestamp(createdAt)
            smallcreatedAtdaydate = datetime.strftime(createdAtdaydate, "%d/%m/%Y")

            try:
                ftFration = (countfollowers / countfollowing)
            except:
                ftFration = 0
            try:
                normalisedfavourites = (favorites / countfollowers)
            except:
                normalisedfavourites = 0
            try:
                normalisedimages = (images / countfollowing)
            except:
                normalisedimages = 0
            try:
                normalisedmentions = (mentions / countfollowing)
            except:
                normalisedmentions = 0
            try:
                normalisedlinks = (links / countfollowing)
            except:
                normalisedlinks = 0
            try:
                normalisedHashtags = (hashtags / countfollowing)
            except:
                normalisedHashtags = 0
            try:
                normalisedRetweets = (retweets / countfollowing)
            except:
                normalisedRetweets = 0
            try:
                normalisedReplies = (replies / countfollowing)
            except:
                normalisedReplies = 0
            if sentiment != int:
                sentiment = 0 #make sure no NaN values are returned for sentiment and breaks code.

            constructed_list = [textlength, normalisedfavourites, normalisedimages, normalisedmentions, normalisedlinks,
                                normalisedHashtags, normalisedRetweets, normalisedReplies, sentiment, originals,
                                publicationscore,
                                userValue, tweetValue, lists, value, actualhashtags, createdAt]
            retdata.append(constructed_list)
    return retdata

def calc_linear_regression_and_importance_coefficient(network_identifier, in_list):
    """
    Authors:
     - Nathan
    Date: 15 2021 July
    Aim:
    Conduct linear regression of features against a target of the normailsedfavourites.

    Secondly, conduct feature importance assessment, i.e. Determine which features are most important to the prediction.

    These attributes will be known as the Campaign Network Attributes  (CNA)
    """

#    NAfile = "C:/Users/Work/Documents/PhD/Part 3/networkattributes.csv"
    col_names = ['textlength', 'normalisedfavourites', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                 'normalisedHashtags', 'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals',
                 'publicationscore', 'userValue', 'tweetValue', 'lists', 'value', 'actualhashtags', 'createdAt']
    # load dataset
#    pima = pd.read_csv("C:/Users/Work/Documents/PhD/Part 3/dump4.csv", header=None, names=col_names)
    pima = pd.DataFrame(in_list, columns=col_names)

    pima.head()
    # define the columbs to be used as indepedant variables


    feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks', 'normalisedHashtags',
                    'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                    'userValue', 'tweetValue', 'lists', 'value']
    # define the features from the columns or independent variables
    X = pima[feature_cols]
    # define the taget variable
    y = pima.normalisedfavourites
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    # define the model
    model = LinearRegression()
    # fit the model
    model.fit(X_train, y_train)
    # make prediction based on model
    y_pred = model.predict(X_test)
    #print('this is y_pred', y_pred)

    # get importance
    importance = model.coef_
    # summarize feature importance
    networkattributes = {}
    for i, v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i, v))
        networkattributes['Feature: %0d' % (i)] = v

    #plot feature importance
    #plt.pyplot.bar([x for x in range(len(importance))], importance)
    # Set number of ticks for x-axis
    #plt.pyplot.xticks(range(len(importance)), feature_cols, rotation='vertical')
    #plt.pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')

    return networkattributes

def calc_decisiontreeregression_and_importance_coefficient(network_identifier, in_list):
    """
    Authors:
     - Nathan
    Date: 15 2021 July
    Aim:
    Conduct Decision Tree Regression of features against a target of the normailsedfavourites.

    Secondly, conduct feature importance assessment, i.e. Determine which features are most important to the prediction.

    These attributes will be known as the Campaign Network Attributes  (CNA)
    """

#    NAfile = "C:/Users/Work/Documents/PhD/Part 3/networkattributes.csv"
    col_names = ['textlength', 'normalisedfavourites', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                 'normalisedHashtags', 'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals',
                 'publicationscore',
                 'userValue', 'tweetValue', 'lists', 'statuses', 'value', 'actualhashtags', 'createdAt']
    # load dataset
#    pima = pd.read_csv("C:/Users/Work/Documents/PhD/Part 3/dump4.csv", header=None, names=col_names)
    pima = pd.DataFrame(in_list, columns=col_names)

    pima.head()
    # define the columbs to be used as indepedant variables


    feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks', 'normalisedHashtags',
                    'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                    'userValue', 'tweetValue', 'lists', 'statuses', 'value']
    # define the features from the columns or independent variables
    X = pima[feature_cols]
    # define the taget variable
    y = pima.normalisedfavourites
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    # define the model
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X_train, y_train)
    # make prediction based on model
    y_pred = model.predict(X_test)
    #print('this is y_pred', y_pred)

    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    networkattributes = {}
    for i, v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i, v))
        networkattributes['Feature: %0d' % (i)] = v

    # plot feature importance
    plt.pyplot.bar([x for x in range(len(importance))], importance)
    # Set number of ticks for x-axis
    plt.pyplot.xticks(range(len(importance)), feature_cols, rotation='vertical')
    plt.pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')

    return networkattributes

def calc_linearRidge_and_importance_coefficient(network_identifier, in_list):
    """
    Authors:
     - Nathan
    Date: 15 2021 July
    Aim:
    Conduct linear Ridge Regression of features against a target of the normailsedfavourites.

    Secondly, conduct feature importance assessment, i.e. Determine which features are most important to the prediction.

    These attributes will be known as the Campaign Network Attributes  (CNA)
    """

#    NAfile = "C:/Users/Work/Documents/PhD/Part 3/networkattributes.csv"
    col_names = ['textlength', 'normalisedfavourites', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                 'normalisedHashtags', 'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals',
                 'publicationscore',
                 'userValue', 'tweetValue', 'lists', 'statuses', 'value', 'actualhashtags', 'createdAt']
    # load dataset
#    pima = pd.read_csv("C:/Users/Work/Documents/PhD/Part 3/dump4.csv", header=None, names=col_names)
    pima = pd.DataFrame(in_list, columns=col_names)

    pima.head()
    # define the columbs to be used as indepedant variables


    feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks', 'normalisedHashtags',
                    'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                    'userValue', 'tweetValue', 'lists', 'statuses', 'value']
    # define the features from the columns or independent variables
    X = pima[feature_cols]
    # define the taget variable
    y = pima.normalisedfavourites
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    # define the model
    model = Ridge(alpha=1.0)
    # fit the model
    model.fit(X_train, y_train)
    # make prediction based on model
    y_pred = model.predict(X_test)
    #print('this is y_pred', y_pred)

    # get importance
    importance = model.coef_
    # summarize feature importance
    networkattributes = {}
    for i, v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i, v))
        networkattributes['Feature: %0d' % (i)] = v

    # plot feature importance
    plt.pyplot.bar([x for x in range(len(importance))], importance)
    # Set number of ticks for x-axis
    plt.pyplot.xticks(range(len(importance)), feature_cols, rotation='vertical')
    plt.pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')

    return networkattributes

def plotting3d(data_in1, data_in2, data_in3):
    """
       Authors:
        - Nathan
       Date: 15 2021 July
       Aim:
       Clusters the rows, hashtag or temporal division of the datasets provided.

       Can cluster two dataset on the one set of axis to compare the outcomes and CNA signitures.
       """


    #X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
    X = data_in1
    Xcomponet = []
    Ycomponet = []
    Zcomponet = []
    clusters = []
    PLOT = go.Figure()
    for i in X.values():
        newX = i[0]
        newY = i[1]
        newZ = i[2]
        Xcomponet.append(newX)
        Ycomponet.append(newY)
        Zcomponet.append(newZ)
        clusters.append(0)

    A = np.array(list(zip(Xcomponet, Ycomponet, Zcomponet, clusters)))

    if use_2_datasets:
        Y = data_in2
        X1componet = []
        Y1componet = []
        Z1componet = []
        clusters1 = []
        PLOT = go.Figure()
        for i in Y.values():
            newX = i[0]
            newY = i[1]
            newZ = i[2]
            X1componet.append(newX)
            Y1componet.append(newY)
            Z1componet.append(newZ)
            clusters1.append(1)
        B = np.array(list(zip(X1componet, Y1componet, Z1componet, clusters1)))
    if use_3_datasets:
        Z = data_in3
        X2componet = []
        Y2componet = []
        Z2componet = []
        clusters2 = []
        PLOT = go.Figure()
        for i in Z.values():
            newX = i[0]
            newY = i[1]
            newZ = i[2]
            X2componet.append(newX)
            Y2componet.append(newY)
            Z2componet.append(newZ)
            clusters2.append(1)
        C = np.array(list(zip(X2componet, Y2componet, Z2componet, clusters2)))
        D = np.concatenate((A, B, C), axis=0)
        df = pd.DataFrame(D, columns=['Feature1', 'Feature2', 'Feature3', "Cluster"])
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.array(df['Feature1'])
        y = np.array(df['Feature2'])
        z = np.array(df['Feature3'])
        ax.scatter(x, y, z, marker="s", c=df["Cluster"], s=40, cmap="RdBu")
        pyplot.show()

    else:
        df = pd.DataFrame(A, columns=['Feature1', 'Feature2', 'Feature3', "Cluster"])
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.array(df['Feature1'])
        y = np.array(df['Feature2'])
        z = np.array(df['Feature3'])
        ax.scatter(x, y, z, marker="s", c=df["Cluster"], s=40, cmap="RdBu")
        pyplot.show()

def scatter_in_2D(data_in1, data_in2, data_in3):
    data = pd.read_csv(data_in1)
    data.head()
    X = data[["normailisedimages", "userValue"]]
    data2 = pd.read_csv(data_in2)
    print(X)
    data2.head()
    X2 = data2[["normailisedimages", "userValue"]]
    print(X2)
    #Centroids = (X.sample(n=K))
    pyplot.scatter(X["normailisedimages"], X["userValue"], marker="s", s=80, cmap="black")
    pyplot.scatter(X2["normailisedimages"], X2["userValue"], marker="s", s=80, cmap="blue")
    #pyplot.scatter(Centroids["normailisedimages"], Centroids["userValue"], c='red')
    pyplot.xlabel('Importance of Image')
    pyplot.ylabel('Twitter generated User Value')
    pyplot.show()

def AIclassification(data_in1):
    print(data_in1)
    df = pd.read_csv(data_in1).dropna()
    print(df['Campaign Type'].value_counts())
    x = df.drop('Campaign Type', axis=1)
    y = df['Campaign Type']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.3, random_state=1)
    sc = StandardScaler()
    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    testX_scaled = scaler.transform(testX)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                            max_iter=300, activation='relu',
                            solver='adam')

    mlp_clf.fit(trainX_scaled, trainY)
    y_pred = mlp_clf.predict(testX_scaled)

    print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

    fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for political campaign Dataset")
    pyplot.show()

    pyplot.plot(mlp_clf.loss_curve_)
    pyplot.title("Loss Curve", fontsize=14)
    pyplot.xlabel('Iterations')
    pyplot.ylabel('Cost')
    pyplot.show()

    print(classification_report(testY, y_pred))

    from hyperopt import hp, tpe, fmin

    # Single line bayesian optimization of polynomial function
    best = fmin(fn=lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x),
                space=hp.normal('x', 4.9, 0.5), algo=tpe.suggest,
                max_evals=2000)
    print(best)

    #param_grid = {
   #     'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    #    'max_iter': [50, 100, 150],
      #  'activation': ['tanh', 'relu'],
     #   'solver': ['sgd', 'adam'],
     #   'alpha': [0.0001, 0.05],
      #  'learning_rate': ['constant', 'adaptive'],
    #}

    #grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1, cv=5)
    #grid.fit(trainX_scaled, trainY)

    #print(grid.best_params_)

    #grid_predictions = grid.predict(testX_scaled)

    #print('Accuracy: {:.2f}'.format(accuracy_score(testY, grid_predictions)))

def separate_list_by_rows(list_in, no_of_rows):
    '''
    Takes the list of lists created in the preprocessing stage and then carves it based on limiting the numbers.
    Returns a list of lists of lists.
    :param list_in:
    :param no_of_rows:
    :return:
    '''
    list_out = []

    # first time needs to be one extra
    counter = -1
    current_run = []
    for ind_record in list_in:
        # end the current row and then start a new one
        if counter >= no_of_rows-1:
            list_out.append(current_run)
            counter = 0
            current_run = []
            current_run.append(ind_record)
        # otherwise just append
        else:
            current_run.append(ind_record)
            counter = counter + 1
    # add the remainder
    list_out.append(current_run)
    return list_out


def separate_list_by_time(list_in):
    '''
    Takes the list of lists created in the preprocessing stage and then carves it based on difference in time.
    Returns a list of lists of lists.
    :param list_in:
    :param no_of_rows:
    :return:
    '''
    list_out = []

    # first time needs to be one extra
    current_run = []
    counter = -1
    first_run = True
    for ind_record in list_in:
        if first_run:
            starttime = ind_record[17]
            first_run = False
        else:
            pass

    for ind_record in list_in:
        postedat = ind_record[17]
        finishtime = starttime + time_delta
        if starttime <= postedat <= finishtime:
            list_out.append(current_run)
            counter = counter + 1
            current_run.append(ind_record)
            # otherwise just append
        else:
            current_run.append(ind_record)
            counter = counter + 1

    list_out.append(current_run)
    print('this is time divided data', list_out)
    return list_out


def separate_by_hashtags(list_in):
    hashtagtracker = []
    hashtagnetwork = {}
    mainsethashtag = {}
    for tweet in list_in:
        usedhashtags = tweet[15]
        for h in usedhashtags:
            if h not in hashtagtracker:
                hashtagtracker.append(h)
                hashtagnetwork["%s" % h] = []
                hashtagnetwork["%s" % h].append(tweet)
            else:
                hashtagnetwork["%s" % h].append(tweet)

    for key in hashtagtracker:
        mainsethashtag[key] = hashtagnetwork["%s" % key]
    return mainsethashtag

if __name__ == '__main__':
    #determining the input files and output files for the various types of routines that can be called
    infile1 = "D:/Datasets/TweetBinder and Other datasets/FIFAWWC.json"
    if use_2_datasets:
        infile2 = "D:/Datasets/TweetBinder and Other datasets/#electionDay.json"
    if use_3_datasets:
        infile3 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/7e3c13ed-3333-4d5b-ab70-ebdb583f466d.json"
    outputclusterfile1 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile1.csv"
    outputclusterfile2 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile2.csv"
    outputclusterfile3 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile3.csv"
    outputclusterfile4 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile4.csv"
    rowoutputfile1 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/rowview1.csv"
    rowoutputfile2 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/rowview2.csv"
    rowoutputfile3 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/rowview3.csv"
    rowoutputfile4 = "D:/Datasets/TweetBinder and Other datasets/Twitter Balakot datasets/rowview4.csv"
    input_json1 = {}
    input_json2 = {}
    separated_preprocessed_list1 = []
    separated_preprocessed_list2 = []
    separated_preprocessed_list3 = []
    storeattributes1 = []
    storeattributes2 = []
    storeattributes3 = []

    with open(infile1, encoding='utf-8') as JSONfile:
        input_json = json.load(JSONfile)
    print('---preprocessing dataset 1--- dataset type', dataset1type)
    if dataset1type == 'TweetBinder':
        preprocessed_list1 = preprocess_to_list_tweetbinderonly(input_json)
    elif dataset1type == 'Nintendo':
        preprocessed_list1 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
    elif dataset1type == 'Eurovision4':
        preprocessed_list1 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
    elif dataset1type == 'Pakelection':
        preprocessed_list1 = preprocess_to_list_Pakelect(input_json)
    else:
        preprocessed_list1 = preprocess_to_list_nonTweetBinder(input_json)

    if use_2_datasets:
        with open(infile2, encoding='utf-8') as JSONfile:
            input_json = json.load(JSONfile)
        print('---preprocessing dataset 2--- dataset type', dataset2type)
        if dataset2type == 'TweetBinder':
            preprocessed_list2 = preprocess_to_list_tweetbinderonly(input_json)
        elif dataset2type == 'Nintendo':
            preprocessed_list2 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
        elif dataset2type == 'Eurovision4':
            preprocessed_list2 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
        elif dataset1type == 'pakelection':
            preprocessed_list2 = preprocess_to_list_Pakelect(input_json)
        else:
            preprocessed_list2 = preprocess_to_list_nonTweetBinder(input_json)

    if use_3_datasets:
        with open(infile3, encoding='utf-8') as JSONfile:
            input_json = json.load(JSONfile)
        print('---preprocessing dataset 3--- dataset type', dataset3type)
        if dataset3type == 'TweetBinder':
            preprocessed_list3 = preprocess_to_list_tweetbinderonly(input_json)
        elif dataset3type == 'Nintendo':
            preprocessed_list3 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
        elif dataset3type == 'Eurovision4':
            preprocessed_list3 = preprocess_to_list_Nintendo_and_Eurovision(input_json)
        elif dataset1type == 'pakelection':
            preprocessed_list3 = preprocess_to_list_Pakelect(input_json)
        else:
            preprocessed_list3 = preprocess_to_list_nonTweetBinder(input_json)

    feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                    'normalisedHashtags',
                    'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                    'userValue', 'tweetValue', 'lists', 'statuses', 'value', 'Campaign Type']
    if use_rows:
        print('Data is divided by', no_of_rows_per_run, 'rows')
        if use_2_datasets:
            if use_3_datasets:
                print('processing three(3) datasets')
            else:
                print('processing two(2) datasets')
        else:
            print('processing one(1) dataset')
        print('_-_-_-_-Processing 1st Dataset _-_-_-_-')
        separated_preprocessed_list1 = separate_list_by_rows(preprocessed_list1, no_of_rows_per_run)
        run_no = 0
        cluster_dict1 = {}
        with open(rowoutputfile4, "w", newline='') as output:
            writer = csv.writer(output)
            writer.writerow(feature_cols)
        for run_in in separated_preprocessed_list1:
            if run_no == 0:
                batchnumber = (0, 'to', no_of_rows_per_run)
            else:
                batchnumber = (run_no * no_of_rows_per_run, 'to', (run_no + 1) * no_of_rows_per_run)
            print('processing', batchnumber)
            timeview1 = []
            net_attributes1 = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}',run_in)
            run_no = run_no + 1
            timeview1.append(net_attributes1['Feature: 1'])  # these features create the subspace
            timeview1.append(net_attributes1['Feature: 6'])
            timeview1.append(net_attributes1['Feature: 0'])
            cluster_dict1[run_no] = timeview1
            with open(rowoutputfile1, "w", newline='', encoding='utf-8') as dump:
                writer = csv.writer(dump)
                writer.writerow(net_attributes1.values())
            with open(rowoutputfile1, "r", newline='', encoding='utf-8') as dump,\
                    open(rowoutputfile4, 'a', newline='') as updateddump:
                reader = csv.reader(dump)
                writer = csv.writer(updateddump)
                for row in reader:
                    row.append(campaigntype1)
                    writer.writerow(row)
            dump.close()
            updateddump.close()

        if use_2_datasets:
            print('_-_-_-_-Processing 2nd Dataset _-_-_-_-')
            cluster_dict2 = {}
            separated_preprocessed_list2 = separate_list_by_rows(preprocessed_list2, no_of_rows_per_run)
            run_no = 0
            for run_in in separated_preprocessed_list2:
                if run_no == 0:
                    batchnumber = (0, 'to', no_of_rows_per_run)
                else:
                    batchnumber = (run_no * no_of_rows_per_run, 'to', (run_no + 1) * no_of_rows_per_run)
                print('processing', batchnumber)
                timeview2 = []
                net_attributes2 = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}',
                                                                                    run_in)
                run_no = run_no + 1
                timeview2.append(net_attributes2['Feature: 1'])  # these features create the subspace
                timeview2.append(net_attributes2['Feature: 6'])
                timeview2.append(net_attributes2['Feature: 0'])
                cluster_dict2[run_no] = timeview2
                with open(rowoutputfile2, "w", newline='', encoding='utf-8') as dump:
                    writer = csv.writer(dump)
                    writer.writerow(net_attributes2.values())
                with open(rowoutputfile2, "r", newline='', encoding='utf-8') as dump, \
                        open(rowoutputfile4, 'a', newline='') as updateddump:
                    reader = csv.reader(dump)
                    writer = csv.writer(updateddump)
                    for row in reader:
                        row.append(campaigntype2)
                        writer.writerow(row)
                dump.close()
                updateddump.close()

        if use_3_datasets:
            print('_-_-_-_-Processing 3rd Dataset _-_-_-_-')
            cluster_dict3 = {}
            separated_preprocessed_list3 = separate_list_by_rows(preprocessed_list3, no_of_rows_per_run)
            run_no = 0
            for run_in in separated_preprocessed_list3:
                if run_no == 0:
                    batchnumber = (0, 'to', no_of_rows_per_run)
                else:
                    batchnumber = (run_no * no_of_rows_per_run, 'to', (run_no + 1) * no_of_rows_per_run)
                print('processing', batchnumber)
                timeview3 = []
                net_attributes3 = calc_decisiontreeregression_and_importance_coefficient(f'{network_identifier}_{run_no}',run_in)
                run_no = run_no + 1
                timeview3.append(net_attributes3['Feature: 1'])  # these features create the subspace
                timeview3.append(net_attributes3['Feature: 6'])
                timeview3.append(net_attributes3['Feature: 0'])
                cluster_dict3[run_no] = timeview3
                with open(rowoutputfile3, "w", newline='', encoding='utf-8') as dump:
                    writer = csv.writer(dump)
                    writer.writerow(net_attributes3.values())
                with open(rowoutputfile3, "r", newline='', encoding='utf-8') as dump, \
                        open(rowoutputfile4, 'a', newline='') as updateddump:
                    reader = csv.reader(dump)
                    writer = csv.writer(updateddump)
                    for row in reader:
                        row.append(campaigntype3)
                        writer.writerow(row)
                dump.close()
                updateddump.close()

        null = {}
        if scatter2D:
            if use_2_datasets:
                if use_3_datasets:
                    scatter_in_2D(outputclusterfile1, outputclusterfile2, outputclusterfile3)
                else:
                    scatter_in_2D(outputclusterfile1, outputclusterfile2, null)
            else:
                scatter_in_2D(outputclusterfile1, null, null)
        if plot3d:
            if use_2_datasets:
                if use_3_datasets:
                    plotting3d(cluster_dict1, cluster_dict2, cluster_dict3)
                else:
                    plotting3d(cluster_dict1, cluster_dict2, null)
            else:
                plotting3d(cluster_dict1, null, null)
        if classify:
            AIclassification(outputclusterfile4)

    elif use_hashtags:
        print('Data separated by', no_of_tweets_per_hashtag, 'hashtags')
        if use_2_datasets:
            if use_3_datasets:
                print('processing three(3) datasets')
            else:
                print('processing two(2) datasets')
        else:
            print('processing one(1) dataset')
        print('_-_-_-_-Processing 1st Dataset _-_-_-_-')
        hashtagtracker = separate_by_hashtags(preprocessed_list1)
        run_no = 0
        cluster_dict1 = {}
        feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                        'normalisedHashtags',
                        'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                        'userValue', 'tweetValue', 'lists', 'value', 'Campaign Type']
        with open(outputclusterfile4, "w", newline='') as output:
            writer = csv.writer(output)
            writer.writerow(feature_cols)
        for run_in1 in hashtagtracker:
            run_in2 = hashtagtracker[run_in1]
            if len(run_in2) >= no_of_tweets_per_hashtag:  # this number can be fine turned to determine the hashtag network size analysed.
                run_no = run_no + 1
                print('running hashtag', run_in1)
                print(len(run_in2))
                cluster_list1 = []
                net_attributes = calc_linear_regression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                # unhash this line for DecisionTree Regression
                #net_attributes1 = calc_decisiontreeregression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                # unhash this line for Linear Ridge regression
                #net_attributes2 = calc_linearRidge_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                cluster_list1.append(net_attributes['Feature: 1'])  # these features create the subspace
                cluster_list1.append(net_attributes['Feature: 6'])
                cluster_list1.append(net_attributes['Feature: 0'])
                cluster_dict1[run_in1] = cluster_list1
                with open(outputclusterfile1, "w", newline='', encoding='utf-8') as dump:
                    writer = csv.writer(dump)
                    writer.writerow(net_attributes.values())
                with open(outputclusterfile1, "r", newline='', encoding='utf-8') as dump, \
                        open(outputclusterfile4, 'a', newline='') as updateddump:
                    reader = csv.reader(dump)
                    writer = csv.writer(updateddump)
                    for row in reader:
                            row.append(campaigntype1)
                            writer.writerow(row)
                dump.close()
                updateddump.close()

        if use_2_datasets:
            print('_-_-_-_-Processing 2nd Dataset _-_-_-_-')
            hashtagtracker2 = separate_by_hashtags(preprocessed_list2)
            run_no = 0
            cluster_dict2 = {}
            for run_in1 in hashtagtracker2:
                run_in2 = hashtagtracker2[run_in1]
                if len(run_in2) >= no_of_tweets_per_hashtag:  # this number can be fine turned to determine the hashtag network size analysed.
                    run_no = run_no + 1
                    print('running hashtag', run_in1)
                    print(len(run_in2))
                    cluster_list2 = []
                    net_attributes2 = calc_linear_regression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                #    unhash this line for DecisionTree Regression
                # net_attributes = calc_decisiontreeregression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                # unhash this line for Linear Ridge regression
                # net_attributes2 = calc_linearRidge_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                    cluster_list2.append(net_attributes2['Feature: 1'])  # these features create the subspace
                    cluster_list2.append(net_attributes2['Feature: 6'])
                    cluster_list2.append(net_attributes2['Feature: 0'])
                    cluster_dict2[run_in1] = cluster_list2
                    with open(outputclusterfile2, "w", newline='', encoding='utf-8') as dump:
                        writer = csv.writer(dump)
                        writer.writerow(net_attributes2.values())
                    with open(outputclusterfile2, "r", newline='', encoding='utf-8') as dump, \
                            open(outputclusterfile4, 'a', newline='') as updateddump:
                        reader = csv.reader(dump)
                        writer = csv.writer(updateddump)
                        for row in reader:
                            row.append(campaigntype2)
                            writer.writerow(row)
                dump.close()
                updateddump.close()

        if use_3_datasets:
            print('_-_-_-_-Processing 3rd Dataset _-_-_-_-')
            hashtagtracker3 = separate_by_hashtags(preprocessed_list3)
            run_no = 0
            cluster_dict3 = {}
            for run_in1 in hashtagtracker3:
                run_in2 = hashtagtracker3[run_in1]
                if len(run_in2) >= no_of_tweets_per_hashtag:  # this number can be fine turned to determine the hashtag network size analysed.
                    run_no = run_no + 1
                    print('running hashtag', run_in1)
                    print(len(run_in2))
                    cluster_list3 = []
                    net_attributes3 = calc_linear_regression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                #    unhash this line for DecisionTree Regression
                # net_attributes = calc_decisiontreeregression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                # unhash this line for Linear Ridge regression
                # net_attributes2 = calc_linearRidge_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                    cluster_list3.append(net_attributes3['Feature: 1'])  # these features create the subspace
                    cluster_list3.append(net_attributes3['Feature: 6'])
                    cluster_list3.append(net_attributes3['Feature: 0'])
                    cluster_dict3[run_in1] = cluster_list3
                    with open(outputclusterfile3, "w", newline='', encoding='utf-8') as dump:
                        writer = csv.writer(dump)
                        writer.writerow(net_attributes3.values())
                    with open(outputclusterfile3, "r", newline='', encoding='utf-8') as dump, \
                            open(outputclusterfile4, 'a', newline='') as updateddump:
                        reader = csv.reader(dump)
                        writer = csv.writer(updateddump)
                        for row in reader:
                            row.append(campaigntype3)
                            writer.writerow(row)
                dump.close()
                updateddump.close()
        null = {}
        if scatter2D:
            if use_2_datasets:
                if use_3_datasets:
                    scatter_in_2D(outputclusterfile1, outputclusterfile2, outputclusterfile3)
                else:
                    scatter_in_2D(outputclusterfile1, outputclusterfile2, null)
            else:
                scatter_in_2D(outputclusterfile1, null, null)
        if plot3d:
            if use_2_datasets:
                if use_3_datasets:
                    plotting3d(cluster_dict1, cluster_dict2, cluster_dict3)
                else:
                    plotting3d(cluster_dict1, cluster_dict2, null)
            else:
                plotting3d(cluster_dict1, null, null)
        if classify:
            AIclassification(outputclusterfile4)

    elif use_entirecampaign:
        print('dataset undivided')
        run_no = 0
        cluster_dict1 = {}
        header = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                        'normalisedHashtags',
                        'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                        'userValue', 'tweetValue', 'lists', 'statuses', 'value', 'Campaign Type']
        with open(outputclusterfile1, "w", newline='') as dump:
            writer = csv.writer(dump)
            writer.writerow(header)
        for run_in1 in preprocessed_list1:
            run_no = run_no + 1
            cluster_list1 = []
            net_attributes1 = calc_linear_regression_and_importance_coefficient(f'{run_in1}_{run_no}', run_in1)
            cluster_list1.append(net_attributes1['Feature: 1'])  # these features create the subspace
            cluster_list1.append(net_attributes1['Feature: 6'])
            cluster_list1.append(net_attributes1['Feature: 0'])
            cluster_dict1[run_no] = cluster_list1
            with open(outputclusterfile1, "a", newline='') as dump:
                writer = csv.writer(dump)
                writer.writerow(net_attributes1.values())
                dump.close()
        null = {}
        #if scatter2D:
            #scatter_in_2D(cluster_dict1, null)
        #if plot3d:
            #plotting3d(net_attributes1, null)
    """
    This code has been developed in consultation with Dr Ben Turnbull of the University Of New South Wales, like always, your support is deeply appreciated. 
 """
