"""
Authors:
 - Nathan
Date: 15 2021 July
Aim:
Quickly checking a JSON if the event distribution change or spike around physical events.
"""
import json as j
import csv
from uuid import uuid4
from datetime import datetime
import time
from pprint import pprint

cumulativeevent = 0
cumulativeimg = 0
cumulativelink = 0
cumulativehash = 0
eventcount = 0

lastdate = []
data = []

infile = "D:/TweetBinder and Other datasets/Twitter Balakot datasets/5dd75c59-233c-47e7-ad25-e493710778fe.json"
outfile = "C:/Users/Work/Documents/PhD/Part 3/dump4.csv"

with open(infile, encoding='utf-8') as JSONfile:
    d = j.load(JSONfile)

with open(outfile, "w", newline='') as myfile:
    myfile.close()



for i in d:
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
    createdAtdaydate = datetime.fromtimestamp(createdAt)
    smallcreatedAtdaydate = datetime.strftime(createdAtdaydate, "%d/%m/%Y")

    try:
        FtFration = (countfollowers / countfollowing)
    except:
        FtFration = 0
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

    with open(outfile, "a", newline='') as myfile:
        writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        data = [textlength, normalisedfavourites, normalisedimages, normalisedmentions, normalisedlinks, normalisedHashtags, normalisedRetweets, normalisedReplies, sentiment, originals, publicationscore,
                    userValue, tweetValue, lists, statuses, value]
        writer.writerow(data)
        myfile.close()



