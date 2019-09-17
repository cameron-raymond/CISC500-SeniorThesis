#!/usr/bin/python
import sys
import preprocessor as p
import emoji
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
	retweet_df = pd.DataFrame(columns=['screen_name','type'])
	for tweet in tweets:
		tweet_df 	= clean_retweet(tweet)
		retweet_df = retweet_df.append(tweet_df, ignore_index=True)
	return retweet_df

def clean_retweet(tweet_obj):
	# cleaned_tweet 	= []
	tweet			= tweet_obj._json
	# raw_text		= emoji.demojize(tweet['full_text'])
	# cleaned_text 	= p.clean(raw_text)
	# cleaned_tweet 	+= [tweet['id'],'retweet', tweet['created_at'],tweet['source'], tweet['full_text'],cleaned_text,tweet['favorite_count'], tweet['retweet_count']]
	# hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
	# cleaned_tweet.append(hashtags) #append hashtags 
	# mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
	# cleaned_tweet.append(mentions) #append mentions
	# cleaned_tweet.append(tweet['user']['screen_name'])
	single_tweet_df = pd.DataFrame([tweet['user']['screen_name'],'retweet'], columns=['screen_name','type'])
	return single_tweet_df

    

if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		get_retweets(username)
