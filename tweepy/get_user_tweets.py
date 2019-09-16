#!/usr/bin/python
import sys
import os
import tweepy
import emoji
from cred import login
from tweet_config import COLS
import pandas as pd
import preprocessor as p
import re #regular expression
import json
import csv

p.set_options(p.OPT.URL,p.OPT.SMILEY)
CONSUMER_KEY    = login['CONSUMER_KEY']
CONSUMER_SECRET = login['CONSUMER_SECRET']
ACCESS_KEY      = login['ACCESS_KEY']
ACCESS_SECRET   = login['ACCESS_SECRET']
print("--- Authorize Twitter; Initialize Tweepy ---")
auth 		= tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api 		= tweepy.API(auth)

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	#authorize twitter, initialize tweepy
	num_tweets	= 3000
	print("--- Return Tweets for {} ---".format(screen_name))
	tweets 		= tweepy.Cursor(api.user_timeline,screen_name=screen_name,count=num_tweets,include_rts=False,tweet_mode='extended')
	timeline_df = pd.DataFrame(columns=COLS)
	print("--- Clean Data ---")
	for tweet in tweets.items():
		if tweet.lang == 'en':
			tweet_df 	= clean_tweet(tweet)
			timeline_df = timeline_df.append(tweet_df, ignore_index=True)
	return timeline_df
		
def clean_tweet(tweet_obj):
	cleaned_tweet 	= []
	tweet			= tweet_obj._json
	raw_text		= emoji.demojize(tweet['full_text'])
	cleaned_text 	= p.clean(raw_text)
	cleaned_tweet 	+= [tweet['id'], tweet['created_at'],tweet['source'], tweet['full_text'],cleaned_text,tweet['favorite_count'], tweet['retweet_count']]
	hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
	cleaned_tweet.append(hashtags) #append hashtags 
	mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
	cleaned_tweet.append(mentions) #append mentions
	cleaned_tweet.append(tweet['user']['screen_name'])
	single_tweet_df = pd.DataFrame([cleaned_tweet], columns=COLS)
	return single_tweet_df

def write_to_file(file_path,new_data):
	data_frame = new_data
	csvFile = open(file_path, 'a' ,encoding='utf-8')
	data_frame.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")

def get_tweets(screen_name):
	file_path = "../data/{}_data.csv".format(screen_name)
	try:
		os.remove(file_path)
	except:
		pass
	tweets_df = get_all_tweets(screen_name)
	write_to_file(file_path,tweets_df)
	print("--- done for {} ---".format(screen_name))

if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		get_tweets(username)
	# get_tweets("liberal_party")
	# get_tweets("JustinTrudeau")
	# get_tweets("CPC_HQ")
	# get_tweets("AndrewScheer")
	# get_tweets("NDP")
	# get_tweets("theJagmeetSingh")
	# get_tweets("ElizabethMay")
	# get_tweets("CanadianGreens")
	# get_tweets("MaximeBernier")
	# get_tweets("peoplespca")

