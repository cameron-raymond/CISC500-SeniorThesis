import sys
import re
import preprocessor as p
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from tweet_config import COLS
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
# pylint: disable=no-member
p.set_options(p.OPT.URL,p.OPT.SMILEY)
stop_words =  stopwords.words('english')
# Remove slogans and popular hashtags that don't mean much
stop_words.extend(['getahead','chooseforward','missionpossible','forwardtogether','initforyou','notasadvertised','elxn43','cdnpoli','ppc','gpc','ppc2019','peoplespca'])
# Remove party names - hard to know whether to remove words like "liberal" or "green" as they are often used in other contexts
stop_words.extend(['elxn43','cdnpoli','ppc','ndp','gpc','pcs','ppc2019','peoplespca'])
stop_words.extend(['get','dont','let','&amp;','amp','canadian']) #some words that aren't in the stopwords list but seem like they should be

def clean_text(sentence):
    sentence = sentence.lower()
    sentence = p.clean(sentence)
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for token in token_words:
        token = re.sub("[,\.!?']", '', token)
        token = lemmatizer.lemmatize(token)
        if token not in stop_words and len(token) >= 3:
            stem_sentence.append(token)
            stem_sentence.append(" ")
    return "".join(stem_sentence)

def clean_tweet(tweet_obj):
    cleaned_tweet 	= []
    tweet			= tweet_obj._json
    raw_text		= emoji.demojize(tweet['full_text'])
    cleaned_text    = clean_text(raw_text)
    cleaned_tweet 	+= [tweet['id'],'tweet', tweet['created_at'],tweet['source'], tweet['full_text'],cleaned_text,tweet['favorite_count'], tweet['retweet_count']]
    hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
    cleaned_tweet.append(hashtags) #append hashtags 
    mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
    cleaned_tweet.append(mentions) #append mentions
    cleaned_tweet.append(tweet['user']['screen_name'])
    single_tweet_df = pd.DataFrame([cleaned_tweet], columns=COLS)
    return single_tweet_df


if __name__ == "__main__":
    # This is a quick and dirty way to take in already outputted csvs and re-clean the text.
    usernames = sys.argv[1:]
    file_path = "../data/{}_data.csv"
    for username in usernames:
        twitter_df = pd.read_csv(file_path.format(username))
        new_clean = twitter_df["original_text"]
        print("--- cleaning text ---")
        new_clean = new_clean.apply(clean_text)
        twitter_df["clean_text"] = new_clean
        print("--- writing {} to file ---".format(username))
        csvFile = open(file_path.format(username), 'w' ,encoding='utf-8')
        twitter_df.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

