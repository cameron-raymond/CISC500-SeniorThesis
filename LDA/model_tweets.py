#!/usr/local/bin/python3
import sys
import pandas as pd
import nltk
from gensim import corpora, models
from gensim.utils import simple_preprocess

class LDA(object):
    def __init__(self,corpus,num_features=100,num_topics=5):
        """

        """
        super().__init__()
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=15, no_above=0.9)
        self.corpus     = [self.dictionary.doc2bow(doc) for doc in corpus]
        self.num_topics = num_topics
        term_freq_inverse_document_freq = models.TfidfModel(self.corpus)
        self.corpus_tfidf = term_freq_inverse_document_freq[self.corpus]
        self.train()

    def train(self):
        lda_model_tfidf = models.LdaMulticore(self.corpus_tfidf, num_topics=self.num_topics, id2word=self.dictionary, passes=2, workers=4)
        lda_model = models.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=2, workers=2)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
    

if __name__ == "__main__":
    usernames = sys.argv[1:]
    frames = []
    for username in usernames:
        file_path = "../data/{}_data.csv".format(username)
        timeline_df = pd.read_csv(file_path)
        print("Number of Tweets for {} is {}".format(username,len(timeline_df)))
        frames.append(timeline_df)
    full_df = pd.concat(frames)
    print("len of combined data frame {}".format(len(full_df)))
    full_df['tokenized'] = full_df.apply(lambda row: nltk.word_tokenize(row['clean_text']), axis=1)
    text_corpus = full_df['tokenized']
    topic_model = LDA(text_corpus)