import numpy as np

num_tweets = 10
config = {
    "usernames": ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"],
    "alphas": np.arange(0,1,0.5),
    "num_tweets": num_tweets,
    "save": False,
    "num_per_alpha": 2,
    "kwargs": {
        "tweet_dist": (num_tweets,num_tweets//5),
        "n": 5,
        "epsilon": 0.9,
        "use_model": False,
        "verbose": False,
    }
}