#!/usr/bin/python
import sys
from get_user_tweets import clean_tweet
from tweet_config import * 
import pandas as pd
import tweepy

# TODO Find way to get more than 100 retweets
def get_retweets(tweet_id):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	#authorize twitter, initialize tweepy
	num_tweets	= 100
	print("--- Return retweets for {} ---".format(tweet_id))
	tweets 		= api.retweets(id=tweet_id,count=num_tweets,tweet_mode='extended')
	retweet_df = pd.DataFrame(columns=COLS)
	for tweet in tweets:
		tweet_df 	= clean_tweet(tweet)
		retweet_df = retweet_df.append(tweet_df, ignore_index=True)
	return retweet_df
    

if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		get_retweets(username)
