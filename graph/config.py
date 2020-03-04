import numpy as np

num_tweets = 10
config = {
    "alphas": np.arange(0,1,0.25),
    "num_tweets": num_tweets,
    "save": True,
    "num_per_alpha": 2,
    "kwargs": {
        "tweet_dist": (num_tweets,num_tweets//5),
        "n": 5,
        "epsilon": 0.9,
        "use_model": False,
        "verbose": False,
    }
}