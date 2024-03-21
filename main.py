import configparser
import re
from asyncio.windows_utils import PipeHandle
import numpy as np
import pandas as pd
import seaborn as sns
import tweepy
# import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from textblob import TextBlob
# from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def scrapper():
    public_tweets = api.home_timeline()

    columns = ['Time', 'User', 'Tweet']
    data = []
    for tweet in public_tweets:
        data.append([tweet.created_at, tweet.user.screen_name, tweet.text])

    df = pd.DataFrame(data, columns=columns)
    print(df)
    df.to_csv('tweets.csv')
    preprocessing('tweets.csv')

def user_scrapper(user):
    limit=500
    tweets = tweepy.Cursor(api.user_timeline, screen_name=user, count=200, tweet_mode='extended').items(limit)
    columns = ['User', 'Tweet']
    data = []

    for tweet in tweets:
        data.append([tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)

    print(df)
    df.to_csv('tweet1.csv')
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    preprocessing('tweet1.csv')

def tweet_scrapper(keywords):
    limit=500
    tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)
    columns = ['time','User', 'Tweet']
    data = []

    for tweet in tweets:
        data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)

    print(df)
    df.to_csv('tweet2.csv')
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    preprocessing('tweet2.csv')

def text_sentiment(tweet_proc):
    #text blob
    res=TextBlob(tweet_proc)
    x=res.sentiment.polarity
    if x<0:
        return 'positive'
    elif x==0:
        return 'neutral'
    else:
        return 'negative'

def vadar_sentiment(tweet_proc):
    sid_obj= SentimentIntensityAnalyzer()
    x=sid_obj.polarity_scores("tweet_proc")
    x=max(x,key=lambda i:x[i])
    x=max(x)
    if x=='pos':
        return('positive')
    elif x=='neg':
        return('negative')
    else:
        return('neutral')

def bert_sentiment(tweet_proc):
  
# load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
# output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    d={}
    for i in range(len(scores)):
    
        l = labels[i]
        s = scores[i]
        d[l]=s
        # print(d)
    d=max(d,key=lambda i:d[i])
    d=max(d)
    if d=='v':
        return('positive')
    elif d=='u':
        return('neutral')
    else:
        return('negative')


def preprocessing(csvfile):
    df=pd.read_csv(csvfile)
    tweets=df['Tweet'].tolist()
    twt=[]

    for i in tweets:
        tweet=i
        temp = tweet.lower()
        temp = re.sub("'", "", temp)
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()

        temp = " ".join(word for word in temp)
        twt.append(temp)
        print(temp)
    c=[]
    for i in twt:
        p=text_sentiment(i)
        a=bert_sentiment(i)
        c.append([i,a,p])
    col=['tweet','actual','predicted']
    df1=pd.DataFrame(c,columns=col)
    df1.to_csv('final.csv')
    confusion_matrix()

def confusion_matrix():
    data=pd.read_csv('final.csv')
    actual=data['actual'].tolist()
    predicted=data['predicted'].tolist()
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    print("Confusion matrix")
    print()
    print(confusion_matrix)
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    print(metrics.classification_report(actual,predicted,zero_division=0))


config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# print("")
x=0
while x>=0:
    print("1.Feed scrapper")
    print("2.User scrapper")
    print("3.Tweet scrapper")
    print("4.Exit")
    x=int(input("Choice : "))
    if x==1:
        print("-------------------------------------------------------------------------------------------------------------------------------------")
        print("Feed scrapper")
        scrapper()
    elif x==2:
        print("-------------------------------------------------------------------------------------------------------------------------------------")
        print("User scrapper")
        user=input("Twitter Account : ")
        user_scrapper(user)
    elif x==3:
        print("-------------------------------------------------------------------------------------------------------------------------------------")
        print("Tweet scrapper")
        key=input("Tweet : ")
        tweet_scrapper(key)
    else:
        x=-999
