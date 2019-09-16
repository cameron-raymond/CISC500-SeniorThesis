#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import re
import collections
from wordcloud import WordCloud, STOPWORDS


class Visualizations:
	def __init__(self,username):
		self.twitter_df = None
		try:
			self.twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
		except:
			pass
	
	def hashtag_chart(self):
		twitter_df = self.twitter_df
		hashtags = []
		hashtag_pattern = re.compile(r"#[a-zA-Z]+")
		hashtag_matches = list(twitter_df['clean_text'].apply(hashtag_pattern.findall))
		hashtag_dict = {}
		for match in hashtag_matches:
			for singlematch in match:
				if singlematch not in hashtag_dict.keys():
					hashtag_dict[singlematch] = 1
				else:
					hashtag_dict[singlematch] = hashtag_dict[singlematch]+1
		hashtag_ordered_list =sorted(hashtag_dict.items(), key=lambda x:x[1])
		hashtag_ordered_list = hashtag_ordered_list[::-1]
		#Separating the hashtags and their values into two different lists
		hashtag_ordered_values = []
		hashtag_ordered_keys = []
		#Pick the 20 most used hashtags to plot
		for item in hashtag_ordered_list[0:20]:
			hashtag_ordered_keys.append(item[0])
			hashtag_ordered_values.append(item[1])
		fig, ax = plt.subplots(figsize = (12,12))
		y_pos = np.arange(len(hashtag_ordered_keys))
		ax.barh(y_pos ,list(hashtag_ordered_values)[::-1], align='center', color = 'green', edgecolor = 'black', linewidth=1)
		ax.set_yticks(y_pos)
		ax.set_yticklabels(list(hashtag_ordered_keys)[::-1])
		ax.set_xlabel("Nº of appereances")
		ax.set_title("Most used #hashtags", fontsize = 20)
		plt.tight_layout(pad=3)
		plt.show()

	def wordcloud(self):
		text = self.twitter_df['clean_text']
		wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black',stopwords = STOPWORDS).generate(str(text))
		fig = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
		plt.imshow(wordcloud, interpolation = 'bilinear')
		plt.axis('off')
		plt.tight_layout(pad=0)
		plt.show()
		

	


if __name__ == '__main__':
    username = sys.argv[1]
    
    # twitter_df.head()
    # justMentions = len([twitter_df['mentions'].notnull()])
    hashtags = []
    hashtag_pattern = re.compile(r"#[a-zA-Z]+")
    hashtag_matches = list(twitter_df['clean_text'].apply(hashtag_pattern.findall))
	# percMentions = justMentions/len(twitter_df)
	# print("% with mentions is {}%".format(percMentions*100))
