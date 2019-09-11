import preprocessor as p

COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'lang',
'favorite_count', 'retweet_count', 
'hashtags', 'original_author',]

preprocessor = p.set_options(p.OPT.NUMBER)