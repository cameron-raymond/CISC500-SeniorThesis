from tweet_config import COLS, api
import preprocessor as p
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import emoji
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
p.set_options(p.OPT.URL,p.OPT.SMILEY)

def clean_tweet(tweet_obj):
    def stem_sentence(sentence):
        token_words=word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for token in token_words:
            token = lemmatizer.lemmatize(token)
            if token not in stopwords.words('english') and len(token) > 3:
                stem_sentence.append(token)
                stem_sentence.append(" ")
        return "".join(stem_sentence)

    cleaned_tweet 	= []
    tweet			= tweet_obj._json
    raw_text		= emoji.demojize(tweet['full_text'])
    cleaned_text 	= p.clean(raw_text)
    cleaned_text    = stem_sentence(cleaned_text)
    cleaned_tweet 	+= [tweet['id'],'tweet', tweet['created_at'],tweet['source'], tweet['full_text'],cleaned_text,tweet['favorite_count'], tweet['retweet_count']]
    hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
    cleaned_tweet.append(hashtags) #append hashtags 
    mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
    cleaned_tweet.append(mentions) #append mentions
    cleaned_tweet.append(tweet['user']['screen_name'])
    single_tweet_df = pd.DataFrame([cleaned_tweet], columns=COLS)
    return single_tweet_df
