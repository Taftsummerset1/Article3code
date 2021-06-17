import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import bs4
import textblob
import re
import pandas as pd
import csv


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

polarityscore = []
subjectivityscore = []
def sentiment(statement):
    x = textblob.TextBlob(str(statement))
    l = x.sentiment.polarity
    y = x.sentiment.subjectivity
    polarityscore.append(l)
    subjectivityscore.append(y)
    if l != 0:
        if l > 0:
            return "Positive"
        elif l < 0:
            return "Negative"
    elif l == 0:
        return "Neutral"


mytweets = pd.read_csv("c:/Users/Work/Documents/botdetect/twitter-bot-detection-main/data/twitter_data.csv", encoding='utf-8')
tweetcontent = mytweets.tweet1_text
#print(tweetcontent)
score = []
for i in tweetcontent:
    try:
        print("\n" + clean_tweet(i) + "\n" + sentiment(clean_tweet(i)))
        score.append(sentiment(clean_tweet(i)))
    except Exception as e:
        continue

with open('new2.csv', 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    fieldnames = ['Tweet', 'sentiment', 'Polarity', 'subjectivity']
    writer.writerows(zip(tweetcontent, score, polarityscore, subjectivityscore))

print("\n\nSummary")
print("=" * 12)
print("Got " + str(len(mytweets)) + " Tweets\n")
print("Positive: " + str(int(score.count("Positive"))))
print("Negative: " + str(int(score.count("Negative"))))
print("Neutral: " + str(int(score.count("Neutral"))))
print("=" * 12)
