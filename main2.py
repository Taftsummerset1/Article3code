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



from matplotlib import pyplot

network_identifier = 'balakot'
dataset1type = 'TweetBinder'
dataset2type = 'TweetBinder'
use_time = False
use_rows = True
use_hashtags = False
no_of_rows_per_run = 100
time_delta = 3600 #in seconds (unix , 3600 = 1 hour, 86400 = day)

cumulativeevent = 0
cumulativeimg = 0
cumulativelink = 0
cumulativehash = 0
eventcount = 0

lastdate = []
data = []

def preprocess_to_list_nonTweetBinder(input_json):
    retdata = []
    for i in input_json:
        mentions = []
        hashtags = []
        urllist = []
        createdAt = i['created_at']
        url = i['entities']['urls']
        retweeted = i['retweeted']
        text = i['text']
        textlength = len(text)
        followers = i['user']['followers_count']
        following = i['user']['friends_count']
        mentionlist = i['entities']['user_mentions']
        hashtaglist = i['entities']['hashtags']
        favoritecount = i['favorite_count']

        if mentionlist:
            for m in mentionlist:
                mentions.append(m['screen_name'])
        if hashtaglist:
            for h in hashtaglist:
                hashtags.append(h['text'])
        if url:
            for u in url:
                urllist.append(u)
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
                            userValue, tweetValue, lists, statuses, value, hashtags, createdAt]
        print(constructed_list)
        retdata.append(constructed_list)
    #print('this is retdata', retdata)
    return retdata


def preprocess_to_list_tweetbinderonly(input_json):
    """
    Authors:
     - Nathan
    Date: 15 2021 July
    Aim:
    Read in the JSON and pull out features of each tweet.
    """
    retdata = []

    for i in input_json:
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
        publicationscore =i['counts']['publicationScore']
        userValue = i['counts']['userValue']
        tweetValue =i['counts']['tweetValue']
        countfollowers = i['user']['followers']
        countfollowing = i['user']['following']
        lists = i['user']['counts']['lists']
        statuses = i['user']['counts']['statuses']
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

        constructed_list = [textlength, normalisedfavourites, normalisedimages, normalisedmentions, normalisedlinks, normalisedHashtags, normalisedRetweets, normalisedReplies, sentiment, originals, publicationscore,
                    userValue, tweetValue, lists, statuses, value, actualhashtags, createdAt]
        retdata.append(constructed_list)
    print('this is retdata', retdata)
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

    # plot feature importance
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
    Conduct linear regression of features against a target of the normailsedfavourites.

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
    #plt.pyplot.bar([x for x in range(len(importance))], importance)
    # Set number of ticks for x-axis
    #plt.pyplot.xticks(range(len(importance)), feature_cols, rotation='vertical')
   # plt.pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')

    return networkattributes

def calc_linearRidge_and_importance_coefficient(network_identifier, in_list):
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
    #plt.pyplot.bar([x for x in range(len(importance))], importance)
    # Set number of ticks for x-axis
    #plt.pyplot.xticks(range(len(importance)), feature_cols, rotation='vertical')
    #plt.pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')

    return networkattributes

def clustering3d(data_in1, data_in2):
    """
       Authors:
        - Nathan
       Date: 15 2021 July
       Aim:
       Cluster the Campaign Network Attributes generated by the importance assessment in importance coefficient function
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
    print(A)

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
    print(B)
    C = np.concatenate((A, B), axis=0)
    print('this is C', C)
    #PLOT.add_trace(go.Scatter3d(x=Xcomponet,
                                   # y=Ycomponet,
                                  #  z=Zcomponet))
    #PLOT.show()

    df = pd.DataFrame(C, columns=['Feature1', 'Feature2', 'Feature3', "Cluster"])

    print(df)


    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    z = np.array(df['Feature3'])

    ax.scatter(x, y, z, marker="s", c=df["Cluster"], s=40, cmap="RdBu")

    pyplot.show()

    # define the model
    #model = DBSCAN()
    # fit the model
    #model.fit(A)
    # assign a cluster to each example
    #yhat = model.fit_predict(A)
    #print('this is yhat', yhat)
    # retrieve unique clusters
    #clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    #for cluster in clusters:
        # get row indexes for samples with this cluster
        #row_ix = where(yhat == cluster)
        # create scatter of these samples
        #pyplot.scatter(A[row_ix, 0], A[row_ix, 1])
    # outputs
    #pyplot.show()
    #pyplot.savefig(f'cluster_{network_identifier}.png')

def plotting3d(data_in):

    X = data_in
    Xcomponet = []
    Ycomponet = []
    Zcomponet = []
    listofdatapoints = []
    for i in X:
        listofdatapoints.append(i)
    labels = np.asarray(listofdatapoints)
    for i in X.values():
        newX = i[0]
        newY = i[1]
        newZ = i[2]
        Xcomponet.append(newX)
        Ycomponet.append(newY)
        Zcomponet.append(newZ)
        zyxcomponet = zip(Xcomponet, Ycomponet, Zcomponet)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xcomponet, Ycomponet, Zcomponet)
    plt.title("3D Scatter plot of CNAs for Hashtag networks")
    plt.show()

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
    print(list_in)
    hashtagtracker = []
    hashtagnetwork = {}
    mainsethashtag = {}
    for tweet in list_in:
        usedhashtags = tweet[16]
        for h in usedhashtags:
            if h not in hashtagtracker:
                hashtagtracker.append(h)
                hashtagnetwork["%s" % h] = []
                hashtagnetwork["%s" % h].append(tweet)
            else:
                hashtagnetwork["%s" % h].append(tweet)

    for key in hashtagtracker:
        mainsethashtag[key] = hashtagnetwork["%s" % key]
    #print('this is mainsethashtag - hashtagtracker', mainsethashtag)
    return mainsethashtag

def cluster_results(list_in):
    """
          Authors:
           - Nathan
          Date: 15 2021 July
          Aim:
          Cluster the Campaign Network Attributes generated by the importance assessment in importance coefficient function
          """

    X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=0)
    data = np.array(list_in)
    # define the model
    model = DBSCAN()
    # fit the model
    model.fit(data)
    # assign a cluster to each example
    yhat = model.fit_predict(data)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # outputs
    pyplot.show()
    pyplot.savefig(f'cluster_{network_identifier}.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    infile1 = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/5dd75c59-233c-47e7-ad25-e493710778fe.json"
    infile2 = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/470f96d4-4fe7-42c5-9f35-6154f592e5f4.json"
    outputclusterfile = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile.csv"
    timeoutputfile1 = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/timeview.csv"
    timeoutputfile2 = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/timeview1.csv"
    input_json1 = {}
    input_json2 = {}
    separated_preprocessed_list1 = []
    separated_preprocessed_list2 = []
    storeattributes1 = []
    storeattributes2 = []

    with open(infile1, encoding='utf-8') as JSONfile:
        input_json = json.load(JSONfile)
    if dataset1type == 'TweetBinder':
        preprocessed_list1 = preprocess_to_list_tweetbinderonly(input_json)
    elif dataset1type != 'TweetBinder':
        preprocessed_list1 = preprocess_to_list_nonTweetBinder(input_json)

    with open(infile2, encoding='utf-8') as JSONfile:
        input_json = json.load(JSONfile)
    if dataset2type == 'TweetBinder':
        preprocessed_list2 = preprocess_to_list_tweetbinderonly(input_json)
    elif dataset2type != 'TweetBinder':
        preprocessed_list2 = preprocess_to_list_nonTweetBinder(input_json)

    cluster_dict1 = {}
    cluster_dict2 = {}
    feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                    'normalisedHashtags',
                    'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                    'userValue', 'tweetValue', 'lists', 'statuses', 'value']
    if use_rows:
        separated_preprocessed_list1 = separate_list_by_rows(preprocessed_list1, no_of_rows_per_run)
        run_no = 0
        with open(timeoutputfile1, "w", newline='', encoding='utf-8') as dump:
            writer = csv.writer(dump)
            writer.writerow(feature_cols)
        for run_in in separated_preprocessed_list1:
            timeview1 = []
            net_attributes1 = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}',run_in)
            run_no = run_no + 1
            timeview1.append(net_attributes1['Feature: 1'])  # these features create the subspace
            timeview1.append(net_attributes1['Feature: 6'])
            timeview1.append(net_attributes1['Feature: 0'])
            cluster_dict1[run_no] = timeview1
            #print(cluster_dict1[run_no])
            #print('this is clusterdict1', cluster_dict1)
            with open(timeoutputfile1, "w", newline='', encoding='utf-8') as dump:
                writer = csv.writer(dump)
                writer.writerow(net_attributes1.values())
                dump.close()

    if use_rows:
        separated_preprocessed_list2 = separate_list_by_rows(preprocessed_list2, no_of_rows_per_run)
        #print('Data separated rows')
        run_no = 0
        for run_in in separated_preprocessed_list2:
            timeview2 = []
            net_attributes2 = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}',run_in)
            run_no = run_no + 1
            timeview2.append(net_attributes2['Feature: 1'])  # these features create the subspace
            timeview2.append(net_attributes2['Feature: 6'])
            timeview2.append(net_attributes2['Feature: 0'])
            cluster_dict2[run_no] = timeview2
            #print(cluster_dict2[run_no])
            #print('this is clusterdict2', cluster_dict2)
            with open(timeoutputfile2, "w", newline='', encoding='utf-8') as dump:
                writer = csv.writer(dump)
                writer.writerow(net_attributes2.values())
                dump.close()

        print('thisi s the right now, not the other one', cluster_dict1, cluster_dict2)
        clustering3d(cluster_dict1, cluster_dict2)

    elif use_hashtags:
        print('Data separated by Hashtag')
        hashtagtracker = separate_by_hashtags(preprocessed_list1)
        #print('thisis hashtag tracker total', hashtagtracker)
        run_no = 0
        #cluster_dict = {"Run number": [], "Hashtags": [], "X axis": [], "Y axis": [], "Z axis": []}
        cluster_dict = {}
        cluster_dict1 = {}
        cluster_dict2 = {}
        feature_cols = ['textlength', 'normailisedimages', 'normalisedmentions', 'normalisedlinks',
                        'normalisedHashtags',
                        'normalisedRetweets', 'normalisedReplies', 'sentiment', 'originals', 'publicationscore',
                        'userValue', 'tweetValue', 'lists', 'statuses', 'value']
        with open(outputclusterfile, "w", newline='') as dump:
            writer = csv.writer(dump)
            writer.writerow(feature_cols)
        for run_in1 in hashtagtracker:
            run_in2 = hashtagtracker[run_in1]
            if len(run_in2) >= 10:  # this number can be fine turned to determine the hashtag network size analysed.
                run_no = run_no + 1
                print('running hashtag', run_in1)
                print(len(run_in2))
                cluster_list = []
                cluster_list1 = []
                cluster_list2 = []
                net_attributes = calc_linear_regression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                net_attributes1 = calc_decisiontreeregression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                net_attributes2 = calc_linearRidge_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                cluster_list.append(run_no)
                #cluster_list.append(run_in1)
                cluster_list.append(net_attributes['Feature: 1'])  # these features create the subspace
                cluster_list.append(net_attributes['Feature: 6'])
                cluster_list.append(net_attributes['Feature: 0'])
                cluster_list1.append(net_attributes['Feature: 1'])  # these features create the subspace
                cluster_list1.append(net_attributes['Feature: 6'])
                cluster_list1.append(net_attributes['Feature: 0'])
                cluster_list2.append(net_attributes['Feature: 1'])  # these features create the subspace
                cluster_list2.append(net_attributes['Feature: 6'])
                cluster_list2.append(net_attributes['Feature: 0'])
                cluster_dict[run_in1] = cluster_list
                cluster_dict1[run_in1] = cluster_list1
                cluster_dict2[run_in1] = cluster_list2
                with open(outputclusterfile, "a", newline='', encoding='utf-8') as dump:
                    writer = csv.writer(dump)
                    writer.writerow(net_attributes.values())
                    dump.close()

            else:
               continue



        #print(cluster_dict)
        #create_subspace_cluster(cluster_dict)
        #clustering3d(cluster_dict)
        #clustering3d(cluster_dict1)
        #clustering3d(cluster_dict2)




    else:
        print('Undivided data used')
        separated_preprocessed_list = [preprocessed_list]
        run_no = 0
        cluster_dict = []
        header = ['images', 'second', 'third']
        with open(outputclusterfile, "w", newline='') as dump:
            writer = csv.writer(dump)
            writer.writerow(header)
        for run_in1 in separated_preprocessed_list:
            net_attributes = calc_linear_regression_and_importance_coefficient(f'{run_in1}_{run_no}', run_in1)
            feature1 = (net_attributes['Feature: 1'])  # these features create the subspace
            feature2 = (net_attributes['Feature: 6'])
            feature3 = (net_attributes['Feature: 0'])
            run_no = run_no + 1
            with open(outputclusterfile, "a", newline='') as dump:
                writer = csv.writer(dump)
                data = [feature1, feature2, feature3]
                writer.writerow(data)
                dump.close()

            #cluster_results(net_attributes)








    """
    Thanks again for the support Ben. I think this is really simple.
 
Preprocessing  - turn the JSON into CSV. This code here outputs the CSV which is used in the second piece of code.

Liner regression (LR) and importance coefficient (IC) – This code here picks up the CSV, applies the LR and then finds most important feature – output is graph and dictionary (networkattributes).
Importantly, this needs to be updated for batch processing. – I want to be able to run X tweets or a block based on time through the LR and IC. This will help with the research itself.
‘createdAtdaydate’ is a variable in 1 that is in unix time if that helps for time filtering.  
Clustering – This code here picks up the dictionary from 2 (networkattributes) and clusters the data.
This will need to be updated to pick up the dictionary, as I haven’t got that far. I think it should be very simple.
 
Further details in the github upload.
 
Thanks again, you’re a legend.
 
N.
 
Nathan Johnson
PhD Student
John Monash Scholar
Chief of Army Scholar
Alliance Plus
Arizona State University
+1 850 460 5414
 """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
