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
from numpy import unique
from numpy import where
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation



from matplotlib import pyplot

network_identifier = 'balakot'
use_time = False
use_rows = False
use_hashtags = True
no_of_rows_per_run = 5
time_delta = 3600 #in seconds (unix , 3600 = 1 hour, 86400 = day)

cumulativeevent = 0
cumulativeimg = 0
cumulativelink = 0
cumulativehash = 0
eventcount = 0

lastdate = []
data = []


def preprocess_to_list(input_json):
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
    #pyplot.bar([x for x in range(len(importance))], importance)
    #print('these are net attributes that resulted in the previous plot', networkattributes)
    #pyplot.show()
    #pyplot.savefig(f'lr_{network_identifier}.png')


    return networkattributes



def create_subspace_cluster(data_in):
    """
       Authors:
        - Nathan
       Date: 15 2021 July
       Aim:
       Cluster the Campaign Network Attributes generated by the importance assessment in importance coefficient function
       """


    #X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
    X = data_in
    Xcomponet = []
    Ycomponet = []
    Zcomponet = []
    PLOT = go.Figure()
    for i in X.values():
        newX = i[0]
        newY = i[1]
        newZ = i[2]
        Xcomponet.append(newX)
        Ycomponet.append(newY)
        Zcomponet.append(newZ)

    A = np.array(list(zip(Xcomponet, Ycomponet, Zcomponet)))
    print(A)

    PLOT.add_trace(go.Scatter3d(x=Xcomponet,
                                    y=Ycomponet,
                                    z=Zcomponet))
    PLOT.show()
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
        #pyplot.Scatter3d(A[row_ix, 0], A[row_ix, 1], A[row_ix, 2])
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
    print('this is mainsethashtag - hashtagtracker', mainsethashtag)
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
    print(X)
    # define the model
    model = DBSCAN()
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.fit_predict(X)
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

    infile = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/7e3c13ed-3333-4d5b-ab70-ebdb583f466d.json"
    outputclusterfile = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/clusterfile.csv"
    input_json = {}
    separated_preprocessed_list = []
    storeattributes = []
    with open(infile, encoding='utf-8') as JSONfile:
        input_json = json.load(JSONfile)
    preprocessed_list = preprocess_to_list(input_json)
    if use_rows:
        separated_preprocessed_list = separate_list_by_rows(preprocessed_list, no_of_rows_per_run)
        print('Data separated rows')
        run_no = 0
        for run_in in separated_preprocessed_list:
            net_attributes = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}', run_in)
            run_no = run_no + 1
            cluster_results(net_attributes)

    elif use_time:
        separated_preprocessed_list = separate_list_by_time(preprocessed_list)
        print('Data separated time')
        run_no = 0
        for run_in in separated_preprocessed_list:
            net_attributes = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}', run_in)
            run_no = run_no + 1
            cluster_results(net_attributes)

    elif use_hashtags:
        print('Data separated by Hashtag')
        hashtagtracker = separate_by_hashtags(preprocessed_list)
        run_no = 0
        #cluster_dict = {"Run number": [], "Hashtags": [], "X axis": [], "Y axis": [], "Z axis": []}
        cluster_dict = {}
        csv_columns = ['Run number', 'Hashtags', 'X axis', 'Y axis', 'Z axis']
        for run_in1 in hashtagtracker:
            run_in2 = hashtagtracker[run_in1]
            if len(run_in2) >= 200:  # this number can be fine turned to determine the hashtag network size analysed.
                print('running hashtag', run_in1)
                print(len(run_in2))
                cluster_list = []
                net_attributes = calc_linear_regression_and_importance_coefficient(f'{run_in2}_{run_no}', run_in2)
                cluster_list.append(net_attributes['Feature: 0']) #these features create the subspace
                cluster_list.append(net_attributes['Feature: 3'])
                cluster_list.append(net_attributes['Feature: 5'])
                cluster_dict[run_in1] = cluster_list
                run_no = run_no + 1
            else:
               continue

        print(cluster_dict)
        #create_subspace_cluster(cluster_dict)
        plotting3d(cluster_dict)




    else:
        print('Undivided data used')
        separated_preprocessed_list = [preprocessed_list]
        run_no = 0
        for run_in in separated_preprocessed_list:
            net_attributes = calc_linear_regression_and_importance_coefficient(f'{network_identifier}_{run_no}', run_in)
            run_no = run_no + 1
            cluster_results(net_attributes)




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
