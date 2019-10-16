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
# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style('whitegrid')



class Visualizations:
	def __init__(self, username):
		self.username = username
		self.file_path = "../visualizations/exploratory_analysis/"
		self.twitter_df = None
		try:
			self.twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
		except:
			pass

	def hashtag_chart(self):
		twitter_df = self.twitter_df
		hashtags = []
		hashtag_pattern = re.compile(r"#[a-zA-Z]+")
		hashtag_matches = list(
		    twitter_df['clean_text'].apply(hashtag_pattern.findall))
		hashtag_dict = {}
		for match in hashtag_matches:
			for singlematch in match:
				if singlematch not in hashtag_dict.keys():
					hashtag_dict[singlematch] = 1
				else:
					hashtag_dict[singlematch] = hashtag_dict[singlematch]+1
		hashtag_ordered_list = sorted(hashtag_dict.items(), key=lambda x: x[1])
		hashtag_ordered_list = hashtag_ordered_list[::-1]
		# Separating the hashtags and their values into two different lists
		hashtag_ordered_values = []
		hashtag_ordered_keys = []
		# Pick the 20 most used hashtags to plot
		for item in hashtag_ordered_list[0:20]:
			hashtag_ordered_keys.append(item[0])
			hashtag_ordered_values.append(item[1])
		fig, ax = plt.subplots(figsize=(12, 12))
		y_pos = np.arange(len(hashtag_ordered_keys))
		ax.barh(y_pos, list(hashtag_ordered_values)[
		        ::-1], align='center', color='green', edgecolor='black', linewidth=1)
		ax.set_yticks(y_pos)
		ax.set_yticklabels(list(hashtag_ordered_keys)[::-1])
		ax.set_xlabel("Nº of appereances")
		ax.set_title("Most used #hashtags", fontsize=20)
		plt.tight_layout(pad=3)
		plt.show()

	def wordcloud(self):
		text = self.twitter_df['clean_text']
		word_cloud = WordCloud(width=2000, height=1000, max_font_size=200, background_color="black", max_words=2000,
                          colormap="nipy_spectral", stopwords=STOPWORDS).generate(str(text))
		plt.figure(figsize=(10, 10))
		plt.imshow(word_cloud, interpolation="hermite")
		plt.axis('off')
		plt.tight_layout(pad=0)
		plt.show()

	def plot_common_words(self):
		title = '{}_Most_Common_Words'.format(self.username)
		# Initialise the count vectorizer with the English stop words
		count_vectorizer = CountVectorizer(stop_words='english')
		# Fit and transform the processed titles
		count_data = count_vectorizer.fit_transform(self.twitter_df['clean_text'])
		words = count_vectorizer.get_feature_names()
		total_counts = np.zeros(len(words))
		for t in count_data:
			total_counts+=t.toarray()[0]
		
		count_dict = (zip(words, total_counts))
		count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
		words = [w[0] for w in count_dict]
		counts = np.array([w[1] for w in count_dict],dtype=np.int)
		x_pos = np.arange(len(words)) 
		
		plt.figure(2, figsize=(15, 15/1.6180))
		plt.subplot(title=title)
		sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
		sns.barplot(x_pos, counts, palette='husl')
		plt.xticks(x_pos, words, rotation=90) 
		plt.xlabel('words')
		plt.ylabel('counts')
		plt.savefig(self.file_path+title, bbox_inches="tight")

		# plt.show()
		

	


if __name__ == '__main__':
	usernames = sys.argv[1:]
	for username in usernames:
		vis_obj = Visualizations(username)
		vis_obj.plot_common_words()
    
    # # twitter_df.head()
    # # justMentions = len([twitter_df['mentions'].notnull()])
    # hashtags = []
    # hashtag_pattern = re.compile(r"#[a-zA-Z]+")
    # hashtag_matches = list(twitter_df['clean_text'].apply(hashtag_pattern.findall))
	# percMentions = justMentions/len(twitter_df)
	# print("% with mentions is {}%".format(percMentions*100))
