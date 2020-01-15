import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# from tweepy.get_user_tweets import write_to_file

topic_num = 5

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #{}: ".format(topic_idx)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def model_document(lda,vectorizer,username):
    """
        Takes in a latent dirichlet allocation, a document term matrix and a data set
        returns, a list with the plurality topic number
    """
    file_path = "../data/{}_data.csv".format(username)
    data_frame = pd.read_csv(file_path)
    clean_text = data_frame["clean_text"].values.astype('U')
    clean_text = vectorizer.transform(clean_text)
    topic_mixture = np.argmax(lda.transform(clean_text),axis=1)
    data_frame["lda_cluster"] = topic_mixture
    return data_frame, file_path


if __name__ == "__main__":
    usernames = sys.argv[1:]
    frames = []
    for username in usernames:
        file_path = "../data/{}_data.csv".format(username)
        timeline_df = pd.read_csv(file_path)
        print("Number of Tweets for {} is {}".format(username,len(timeline_df)))
        frames.append(timeline_df)
    text_data = pd.concat(frames)["clean_text"].values.astype('U')
    tf_idf_vectorizer = TfidfVectorizer(ngram_range = (1,1),min_df = 15, max_df = 0.90).fit(text_data)

    #create document term matrix
    train_dtm = tf_idf_vectorizer.transform(text_data)
    lda = LatentDirichletAllocation(n_components = topic_num)
    lda = lda.fit(train_dtm)
    lda_weights = lda.transform(train_dtm)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_idf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 20)
    for username in usernames:
        added_cluster, file_path = model_document(lda,tf_idf_vectorizer,username)
        csvFile = open(file_path, 'w' ,encoding='utf-8')
        added_cluster.to_csv(csvFile, mode='w', index=False, encoding="utf-8")
        print(added_cluster.head())
