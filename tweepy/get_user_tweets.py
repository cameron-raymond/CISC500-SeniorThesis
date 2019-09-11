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

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	#authorize twitter, initialize tweepy
	print("authorize twitter, initialize tweepy")
	auth 		= tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
	api 		= tweepy.API(auth)
	num_tweets	= 4000
	tweets 		= tweepy.Cursor(api.user_timeline,screen_name=screen_name,count=num_tweets,include_rts=False)
	timeline_df = pd.DataFrame(columns=COLS)
	for tweet in tweets.items():
		if tweet.lang == 'en':
			tweet_df = clean_tweet(tweet)
			timeline_df = timeline_df.append(tweet_df, ignore_index=True)

	return timeline_df
		
		
def clean_tweet(tweet_obj):
	cleaned_tweet 	= []
	tweet			= tweet_obj._json
	raw_text		= emoji.demojize(tweet['text'])
	cleaned_text 	= p.clean(raw_text)
	cleaned_tweet 	+= [tweet['id'], tweet['created_at'],tweet['source'], tweet['text'],cleaned_text, tweet['lang'],tweet['favorite_count'], tweet['retweet_count']]
	hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
	cleaned_tweet.append(hashtags) #append the hashtags
	cleaned_tweet.append(tweet['user']['screen_name'])
	single_tweet_df = pd.DataFrame([cleaned_tweet], columns=COLS)
	return single_tweet_df

def write_to_file(file_path,new_data):
	data_frame = new_data
	# print(data_frame.to_string())
	# if os.path.exists(file_path):
	# 	data_frame = pd.read_csv(file_path,header=0)
	# 	data_frame.append(new_data)
	
	csvFile = open(file_path, 'a' ,encoding='utf-8')
	data_frame.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")

if __name__ == '__main__':
	lib_tweets = "../data/liberal_party_data.csv"
	trudeau_tweets = "../data/trudeau_data.csv"
	try:
		os.remove(lib_tweets)
		os.remove(trudeau_tweets)
	except:
		pass

	#pass in the username of the account you want to download
	lib_df = get_all_tweets("liberal_party")
	trudeau_df = get_all_tweets("JustinTrudeau")
	write_to_file(lib_tweets,lib_df)
	write_to_file(trudeau_tweets,trudeau_df)

