from cred import login
import tweepy

COLS = ['id', 'created_at', 'source', 'original_text','clean_text',
'favorite_count', 'retweet_count', 
'hashtags','mentions', 'original_author',]


CONSUMER_KEY    = login['CONSUMER_KEY']
CONSUMER_SECRET = login['CONSUMER_SECRET']
ACCESS_KEY      = login['ACCESS_KEY']
ACCESS_SECRET   = login['ACCESS_SECRET']
print("--- Authorize Twitter; Initialize Tweepy ---")
auth 		= tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api 		= tweepy.API(auth)