#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 23:45:13 2022

@author: vinaysammangi
"""
import flask
from flask import *
import json
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import random
import string
import pickle
import tweepy
import nltk
import re
nltk.download('all')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask_cors import CORS, cross_origin
from ta import add_all_ta_features
import os
from datetime import datetime
from pytz import timezone
import time
import pytz
import streamlit as st

pkl_folder = "StockpriceForecasting/pkl/"
ticker_pattern = re.compile(r'(^\$[A-Z]+|^\$ES_F)')
ht_pattern = re.compile(r'#\w+')

charonly = re.compile(r'[^a-zA-Z\s]')
handle_pattern = re.compile(r'@\w+')
emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
url_pattern = re.compile(
    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pic_pattern = re.compile('pic\.twitter\.com/.{10}')
special_code = re.compile(r'(&amp;|&gt;|&lt;)')
tag_pattern = re.compile(r'<.*?>')

STOPWORDS = set(stopwords.words('english')).union(
    {'rt', 'retweet', 'RT', 'Retweet', 'RETWEET'})

lemmatizer = WordNetLemmatizer()

def hashtag(phrase):
    return ht_pattern.sub(' ', phrase)

def remove_ticker(phrase):
    return ticker_pattern.sub('', phrase)
    
def specialcode(phrase):
    return special_code.sub(' ', phrase)

def emoji(phrase):
    return emoji_pattern.sub(' ', phrase)

def url(phrase):
    return url_pattern.sub('', phrase)

def pic(phrase):
    return pic_pattern.sub('', phrase)

def html_tag(phrase):
    return tag_pattern.sub(' ', phrase)

def handle(phrase):
    return handle_pattern.sub('', phrase)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    
    # DIS, ticker symbol of Disney, is interpreted as the plural of "DI" 
    # in WordCloud, so I converted it to Disney
    phrase = re.sub('DIS', 'Disney', phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"(he|He)\'s", "he is", phrase)
    phrase = re.sub(r"(she|She)\'s", "she is", phrase)
    phrase = re.sub(r"(it|It)\'s", "it is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"(\'ve|has)", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def onlychar(phrase):
    return charonly.sub('', phrase)

def remove_stopwords(phrase):
    return " ".join([word for word in str(phrase).split()\
                     if word not in STOPWORDS])

def tokenize_stem(phrase):   
    tokens = word_tokenize(phrase)
    stem_words =[]
    for token in tokens:
        word = lemmatizer.lemmatize(token)
        stem_words.append(word)        
    buf = ' '.join(stem_words)    
    return buf

def arrange_text(ds):
    ds['text'] = ds['text'].str.strip().str.lower()
    ds['text'] = ds['text'].apply(emoji)
    ds['text'] = ds['text'].apply(handle)
    ds['text'] = ds['text'].apply(specialcode)
    ds['text'] = ds['text'].apply(hashtag)
    ds['text'] = ds['text'].apply(url)
    ds['text'] = ds['text'].apply(pic)
    ds['text'] = ds['text'].apply(html_tag)
    ds['text'] = ds['text'].apply(onlychar)
    ds['text'] = ds['text'].apply(decontracted)
    ds['text'] = ds['text'].apply(onlychar)
    ds['text'] = ds['text'].apply(tokenize_stem)
    ds['text'] = ds['text'].apply(remove_stopwords)
    return ds

@st.cache(allow_output_mutation=True)
def get_picklefile_stock(path):
    return pickle.load(open(pkl_folder+path, 'rb'))

def get_tweet_sentiments(ds):
    vec = get_picklefile_stock("sentiment/vectorizer.pkl")
    nb_model = get_picklefile_stock("sentiment/sentiment_logistic_cls.pkl")
    sentiments = []
    for text in ds["text"]:
        sentiments.append(nb_model.predict(vec.transform([text]))[0])
    ds["Sentiment"] = sentiments
    return ds

def get_technical_forecasts(company_data,tickerSymbol):
    company_data.columns = ["time","open","high","low","close","volume"]
    company_data_ta = add_all_ta_features(company_data,open="open",high="high",low="low",close="close",volume="volume")
    del company_data_ta["trend_psar_up"]
    del company_data_ta["trend_psar_down"]
    del company_data_ta["time"]
    company_data_ta = company_data_ta.ffill().bfill().tail(1)
    last_close = list(company_data_ta['close'])[0]
    scaler = get_picklefile_stock("scalers/scaler_"+tickerSymbol+".pkl")
    #company_data_ta = scaler.transform(company_data_ta.to_numpy())
    company_data_ta = scaler.transform(company_data_ta)

    reg_models = [get_picklefile_stock(tickerSymbol+"/reg/"+x) for x in os.listdir(pkl_folder+tickerSymbol+"/reg") if x.endswith(".pkl")]
    reg_price_forecasts = []
    reg_price_directions = []
    for reg_model in reg_models:
        forecast = reg_model.predict(company_data_ta)[0]
        reg_price_forecasts.append(forecast)
        if forecast>last_close:
            reg_price_directions.append("Increase")
        else:
            reg_price_directions.append("Decrease")

    cls_models = [get_picklefile_stock(tickerSymbol+"/cls/"+x) for x in os.listdir(pkl_folder+tickerSymbol+"/cls") if x.endswith(".pkl")]
    cls_price_directions = []    
    for cls_model in cls_models:
        forecast = cls_model.predict(company_data_ta)[0]
        cls_price_directions.append(forecast)

    all_directions = cls_price_directions+reg_price_directions
    all_directions = np.array(all_directions)
    reg_price_directions = np.array(reg_price_directions)
    cls_price_directions = np.array(cls_price_directions)
    reg_price_forecasts = np.array(reg_price_forecasts)
    c1 = sum(all_directions=="Increase")
    c2 = sum(all_directions=="Decrease")
    if c1<c2:
        if c2==3:
            confidence = 60
        elif c2==4:
            confidence = 75
        else:
            confidence = 90
            
    if c2<c1:
        if c1==3:
            confidence = 60
        elif c1==4:
            confidence = 75
        else:
            confidence = 90
    forecast = np.mean(reg_price_forecasts)
    return {"Forecast":forecast,"ForecastConfidence":confidence}    

def technical_forecast(tickerSymbol):
    tickerData = yf.Ticker(tickerSymbol) # Get ticker data
    
    timeZ_Ny = pytz.timezone('America/New_York')
    dt_Ny = datetime.now(timeZ_Ny)
    end_date = pd.to_datetime(dt_Ny.strftime('%Y-%m-%d %H:%M:%S'))
    
    tickerDf_30m = tickerData.history(interval='30m', end=end_date).reset_index(drop=False) #get the historical prices for this ticker
    tickerDf_30m["Datetime"] = pd.to_datetime(pd.to_datetime(tickerDf_30m["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S'))
    tickerDf_30m.drop(tickerDf_30m.tail(1).index,inplace=True)    
    tickerDf_30m =  tickerDf_30m[["Datetime","Open","High","Low","Close","Volume"]]
    
    latest_yftime = tickerDf_30m.tail(1).Datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
    latest_yftime = list(latest_yftime)[0]
    latest_yfhour = int(latest_yftime.split('-')[-1].split(' ')[-1].split(':')[0])
    latest_yfmin = int(latest_yftime.split('-')[-1].split(' ')[-1].split(':')[1])
    latest_yfdate = latest_yftime.split(" ")[0]
    tickerDf_30m['Datetime'] = tickerDf_30m['Datetime'].astype(str)
    tz = timezone('US/Eastern')

    current_time = pd.to_datetime(datetime.now(tz)).strftime('%Y-%m-%d %H:%M:%S')
    current_hour = int(current_time.split('-')[-1].split(' ')[-1].split(':')[0])
    current_min = int(current_time.split('-')[-1].split(' ')[-1].split(':')[1])
    current_date = current_time.split(" ")[0]

    temp = get_technical_forecasts(tickerDf_30m.copy(),tickerSymbol)
    if (temp["Forecast"]-list(tickerDf_30m.tail(1).Close)[0])/list(tickerDf_30m.tail(1).Close)[0] > 0.015:
        temp["Forecast"] = list(tickerDf_30m.tail(1).Close)[0]*1.015
    if (list(tickerDf_30m.tail(1).Close)[0]-temp["Forecast"])/list(tickerDf_30m.tail(1).Close)[0] > 0.015:
        temp["Forecast"] = list(tickerDf_30m.tail(1).Close)[0]*0.985
    # if current_hour<10 or current_hour>16 or (current_hour==15 and current_min>29):
    #     temp["Forecast"] = -1000
    # if current_date!=latest_yfdate:
    #     temp["Forecast"] = -1000
    
    data_set = {'Ticker':tickerSymbol,'DataFrame': tickerDf_30m.to_dict('list'),'Forecast':temp["Forecast"],'ForecastConfidence':temp["ForecastConfidence"]}    
    json_dump = json.dumps(data_set)
    #data_set = flask.jsonify(data_set)
    return json_dump

def sentiment_forecast(tickerSymbol):
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAP1UbAEAAAAACYF7EUp%2BeaZn4LtLBcyvK2dESlU%3DuBW4d2x1oRc32va7VWlEmWkHjKqlWJbXWLoQbhBOQP5tbuqyzP'
    client = tweepy.Client(bearer_token)
    response = client.search_all_tweets('#'+tickerSymbol+" stock price"+" OR "+'#'+tickerSymbol+" share price"+" OR "+'$'+tickerSymbol+" stock price"+" OR "+'$'+tickerSymbol+" share price", max_results=50,tweet_fields=['lang', 'created_at'])
      
    company_tweets = []
    for tweet in response.data:
        if tweet.lang=="en":
            company_tweets.append(tweet.text)

    company_tweets = list(set(company_tweets))
    tweets_data = pd.DataFrame(company_tweets)
    tweets_data.columns = ["text"]
    tweets_data["orginal_tweet"] = tweets_data["text"]
    tweets_data = arrange_text(tweets_data)
    # tweets_data.drop_duplicates(inplace=True)
    tweets_data = get_tweet_sentiments(tweets_data)
    tweets_data.columns = ["Tweet","orginal_tweet","Sentiment"]
    tweets_data = tweets_data.loc[tweets_data["Tweet"]!="",]
    bullish_counts = sum(tweets_data["Sentiment"]=="Bullish")
    bearish_counts = sum(tweets_data["Sentiment"]=="Bearish")
    del tweets_data["Tweet"]
    tweets_data.columns = ["Tweet","Sentiment"]
    forecast = "Neutral"
    confidence = 50
    if bullish_counts>bearish_counts:
        forecast = "Bullish"
        confidence = round(bullish_counts/(bullish_counts+bearish_counts),2)*100
    elif bullish_counts<bearish_counts:
        forecast = "Bearish"
        confidence = round(bearish_counts/(bullish_counts+bearish_counts),2)*100
    data_set = {'Ticker':tickerSymbol,'DataFrame': tweets_data.to_dict('list'),'Forecast':forecast,'ForecastConfidence':confidence}
    json_dump = json.dumps(data_set)   
    #data_set = flask.jsonify(data_set)
    return json_dump
