#!/usr/local/bin/python3
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

topic_num = 10

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #{}: ".format(topic_idx+1)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

if __name__ == "__main__":
    usernames = sys.argv[1:]
    frames = []
    for username in usernames:
        file_path = "../data/{}_data.csv".format(username)
        timeline_df = pd.read_csv(file_path)
        print("Number of Tweets for {} is {}".format(username,len(timeline_df)))
        frames.append(timeline_df)
    text_data = pd.concat(frames)["clean_text"].values.astype('U')
    vec_count = CountVectorizer(ngram_range = (1,1),min_df = 15, max_df = 0.90).fit(text_data)

    #create document term matrix
    train_dtm = vec_count.transform(text_data)
    lda = LatentDirichletAllocation(n_components = topic_num)
    lda = lda.fit(train_dtm)
    lda_weights = lda.transform(train_dtm)
    print("\nTopics in LDA model:")
    tf_feature_names = vec_count.get_feature_names()
    print_top_words(lda, tf_feature_names, 20)
