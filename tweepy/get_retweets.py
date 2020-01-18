import sys
import os
import tqdm
from get_user_tweets import write_to_file
from tweet_config import * 
import pandas as pd
import tweepy

class Retweet_Grabber(object):
	def __init__(self, screen_name, *args, **kwargs):
		self.screen_name = screen_name	
		self.file_path =  "../data/{}_retweets.csv".format(screen_name)
		self.tweet_ids, self.retweet_df, self.num_tweets = self.get_old_retweets()	

	def get_old_retweets(self):
		tweet_file_path		= "../data/{}_data.csv".format(self.screen_name)
		tweet_df = pd.read_csv(tweet_file_path)
		exists = os.path.exists(self.file_path)
		old_retweets = pd.DataFrame(columns=RETWEET_COLS)
		if exists:
			old_retweets = pd.read_csv(self.file_path)	
			tweet_df = tweet_df[(~tweet_df["id"].isin(old_retweets["original_tweet_id"])) & tweet_df["retweet_count"]!=0]
			print("--- {} retweets already collected, only collecting {} now ---".format(len(old_retweets["original_tweet_id"].unique()),tweet_df.shape[0]))		
		num_tweets = tweet_df.shape[0]
		return tweet_df,old_retweets,num_tweets

	def put_tweets(self):
		screen_name = self.screen_name
		self.get_user_retweets()
		# assert self. TODO assert so that number of unique "original_tweet_id"s is == to the number of tweets in the df (- the ones that don't have any retweets)
		write_to_file(self.file_path,self.retweet_df)
		print("--- done for {} ---".format(screen_name))

	def get_user_retweets(self):
		screen_name = self.screen_name
		index = 1
		pbar = tqdm.tqdm(total=len(self.tweet_ids))
		for _, row in self.tweet_ids.iterrows():
			tweet_id = row['id']
			retweets = self.get_retweets(tweet_id)
			self.retweet_df = self.retweet_df.append(retweets)
			if index % (self.num_tweets//10) == 0:
				print("\t> writing tweets")
				write_to_file(self.file_path,self.retweet_df)
			index += 1
			pbar.update(1)
		pbar.close()
		self.retweet_df.drop(self.retweet_df.loc[self.retweet_df['original_author']==screen_name].index, inplace=True)
	
	# TODO Find way to get more than 100 retweets
	def get_retweets(self,tweet_id):
		#Twitter only allows access to a users most recent 3240 tweets with this method
		#authorize twitter, initialize tweepy
		tweets 		= api.retweets(id=tweet_id,tweet_mode='extended')
		retweet_df = pd.DataFrame(columns=RETWEET_COLS)
		for tweet in tweets:
			tweet_df 	= self.clean_retweet(tweet,tweet_id)
			retweet_df = retweet_df.append(tweet_df, ignore_index=True)
		return retweet_df

	def clean_retweet(self,tweet_obj,tweet_id):
		cleaned_tweet 	= []
		tweet			= tweet_obj._json
		cleaned_tweet 	+= [tweet_id,tweet['id'],'retweet', tweet['created_at'],tweet['source'],tweet['favorite_count'], tweet['retweet_count']]
		cleaned_tweet.append(tweet['user']['screen_name'])
		single_tweet_df = pd.DataFrame([cleaned_tweet], columns=RETWEET_COLS)
		return single_tweet_df

if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		print("--- starting data collection for {}".format(username))
		user = Retweet_Grabber(username)
		user.put_tweets()

