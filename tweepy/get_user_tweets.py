#!/usr/local/bin/python3
import sys
import os
from datetime import datetime
import tweepy
from tweet_config import COLS, api
from text_cleaning import clean_tweet
import pandas as pd
import json
import csv


def get_tweets(screen_name,most_recent_date=None,en_only=True):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	#authorize twitter, initialize tweepy
	num_tweets	= 3000
	print("--- Return Tweets for {} ---".format(screen_name))
	tweets 		= tweepy.Cursor(api.user_timeline,screen_name=screen_name,count=num_tweets,include_rts=False,tweet_mode='extended')
	timeline_df = pd.DataFrame(columns=COLS)
	print("--- Clean Data ---")
	for tweet in tweets.items():
		if (en_only and tweet.lang == 'en') or not en_only:
			tweet_df 	= clean_tweet(tweet)
			timeline_df = timeline_df.append(tweet_df, ignore_index=True)
	if most_recent_date:
		print("--- Removing Tweets Before {}---".format(most_recent_date))
		timeline_df['to_date'] = pd.to_datetime(timeline_df['created_at']).dt.tz_convert(None)
		timeline_df = timeline_df[timeline_df['to_date'] > most_recent_date]
		timeline_df.drop('to_date',axis=1)
	return timeline_df
		

def write_to_file(file_path,new_data):
	"""
		test
	"""
	data_frame = new_data
	csvFile = open(file_path, 'w' ,encoding='utf-8')
	data_frame.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

def put_tweets(screen_name,en_only=True):
	file_path = "../data/{}_data.csv".format(screen_name)
	exists = os.path.exists(file_path)
	most_recent_date 	= None
	old_timeline 		= None
	if exists:
		old_timeline = pd.read_csv(file_path)
		dates = pd.to_datetime(old_timeline['created_at']).dt.tz_convert(None)
		most_recent_date =dates.max()
		tweets_df = get_tweets(screen_name,most_recent_date,en_only)
		new_tweets = tweets_df.shape[0]
		tweets_df = tweets_df.append(old_timeline,sort=False)
		print("--- Combined timeline shape: {}, added {} tweets---".format(tweets_df.shape,new_tweets))
		write_to_file(file_path,tweets_df)
	else:
		tweets_df = get_tweets(screen_name,most_recent_date,en_only)
		write_to_file(file_path,tweets_df)
	print("--- done for {} ---".format(screen_name))

if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		put_tweets(username,en_only=False)
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

