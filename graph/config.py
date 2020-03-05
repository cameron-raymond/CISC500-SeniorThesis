import numpy as np

num_tweets = 150
config = {
    "usernames": ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"],
    "alphas": np.round(np.arange(0,1.01,0.05),2),
    "num_tweets": num_tweets,
    "save": True,
    "num_per_alpha": 4,
    "kwargs": {
        "tweet_dist": (num_tweets,num_tweets//5),
        "n": 5,
        "epsilon": 0.9,
        "use_model": True,
        "verbose": False,
    }
}