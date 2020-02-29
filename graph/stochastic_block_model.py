import networkx as nx
import numpy as np
from numpy.random import normal, random
import tensorflow as tf
from tensorflow.keras.models import load_model
from build_graph import return_colour, return_legend
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from build_graph import Graph
from matplotlib.lines import Line2D
import sys
import tqdm
from centrality_measures import centrality_per_topic, plot_dual_centralities
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def sample_from_histogram(histogram,n=100):
    """ 
        Creates a sample of retweet counts based on the histogram given.
        Parameters
        ----------
        :param histogram:
            The bin and count values that denote the frequency of different retweet counts.
    """
    hist,bins = histogram
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    return np.round(random_from_cdf).astype(int)

def possible_tweets(G, aUser, topic=None, leader=None):
    """
        Returns all tweet nodes of a certain topic that have not already been retweeted by a user.
    """
    # assert (topic is None or leader is None) and not (topic is None and leader is None), "Only one of the topic and leader can be present"
    possible_tweets = []
    for node in G.nodes():
        attributes = G.nodes[node]
        already_connected = node in G[aUser]
        is_remaining_tweet = attributes["type"] == "tweet" and not already_connected
        # If we're building a hybrid graph, only return leader x's tweets that pertain to topic i
        if topic is not None and leader is not None:
            if is_remaining_tweet and attributes["lda_cluster"] == topic and leader in G[node]:
                possible_tweets.append(node)
        elif topic is not None:
            if is_remaining_tweet and attributes["lda_cluster"] == topic:
                possible_tweets.append(node)
        elif leader is not None:
            if is_remaining_tweet and leader in G[node]:
                possible_tweets.append(node)
    return possible_tweets

def predict_next_retweet(history, model, use_model=True):
    next_decision_distribution = history + np.full(history.shape, 1)
    if use_model:
        history = history.reshape(1, len(history))
        next_decision_distribution = model.predict(
            history).reshape(history.shape[1])
    # Squish it using softmax to make it a probability distribution.
    return softmax(next_decision_distribution)

def init_graph(n=5, tweet_dist=(1000, 300), k=7, m=60000,verbose=False):
    topics = [i for i in range(k)]
    G = nx.Graph()
    for user in range(n):
        G.add_node(user, type='user')

    if verbose: print("--- Adding users and tweets ---")
    num_tweets = [max(int(normal(loc=tweet_dist[0], scale=tweet_dist[1])), 1) for _ in range(n)]
    total_nodes = sum(num_tweets)+m
    pbar = tqdm.tqdm(total=total_nodes) if verbose else None
    for user,num_tweets in enumerate(num_tweets):
        topic_distribution = softmax([random() for i in range(k)])
        # here, a= are the topic numbers [0...6] and p is the probability of choosing tweet a
        tweets = np.random.choice(a=topics, size=num_tweets, p=topic_distribution)
        for tweet, topic in enumerate(tweets):
            node = "{}_{}".format(user, tweet)
            G.add_node(node, type="tweet", lda_cluster=topic)
            G.add_edge(user, node)
            if verbose: pbar.update(1)

    # initialize the probability array
    topic_history = np.zeros(k)
    leader_history = np.zeros(n)
    for new_user in range(m):
        # add a new user
        username = "user_{}".format(new_user)
        G.add_node(username, type="retweet",topic_history=topic_history, leader_history=leader_history)
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    return G

def draw_graph(G, save=False, file_name="stochastic_block_graph", file_type='png', transparent=False, title="Stochastic Block Model"):
    """
    Handles rendering and drawing the network.
    Parameters
    ----------
    :param G: `optional` 
    
    a networkx graph. If present draws this graph instead of the one built in the constructor.

    :param save: `optional` 
    
    A boolean. If true saves an image of the graph to `/visualizations` otherwise renders the graph.

    :param file_type: `optional` 
    
    A string. If save flag is true it saves graph with this file extension.

    :param transparent: `optional` 
    
    A Boolean. If true it only renders the tweet's; no edges or user nodes.
    """
    print("--- Adding colours and labels ---")
    colors = []
    legend = set()
    labels = {}
    pbar = tqdm.tqdm(total=len(G.nodes()))
    for node in G.nodes():
        attributes = G.nodes[node]
        if 'type' in attributes:
            if attributes['type'] == 'retweet':
                colors.append(
                    '#79bfd3ff') if not transparent else colors.append("white")
            elif attributes['type'] == 'tweet':
                cluster = return_colour(attributes["lda_cluster"])
                legend.add(cluster)
                colors.append(cluster[0])
            elif attributes['type'] == 'user':
                labels[node] = node
                colors.append(
                    'red') if not transparent else colors.append("white")
        pbar.update(1)
    pbar.close()
    plt.figure(figsize=(30, 30))
    pos = graphviz_layout(G, prog="sfdp")
    print("--- Drawing {} nodes and {} edges ---".format(len(G.nodes()),
                                                         G.number_of_edges()))
    width = 0.2 if not transparent else 0
    # node_size = 20 if not transparent else 200
    nx.draw_networkx(G, pos,
                     node_color=colors,
                     with_labels=False,
                     alpha=0.75,
                     node_size=200,
                     width=width,
                     arrows=False
                     )
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color='r')
    plt.legend(handles=return_legend(legend), loc="best")
    plt.title(title, fontdict={'fontsize': 30})
    # Turn off borders
    plt.box(False)
    plt.savefig("../visualizations/random_graphs/{}.{}".format(file_name,file_type),bbox_inches="tight") if save else plt.show()

def stochastic_topic_graph(n=5, tweet_dist=(1000, 300), k=7, m=60000,retweet_histogram=None, epsilon=0.95,use_model=True,verbose=False,**kwargs):
    """
        Build a stochastic block model based off of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param posteriors: 
        
        A list of prior observations as to how retweeters have behaved. 

        :param n: 
        
        The number of "party leaders"

        :param tweet_dist: 
        
        A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: 
        
        The number of topics that a tweeter might tweet about.

        :param m: 
        
        The max number of retweeters in the system. 

        :param p: 
        
        The default probability of a retweeter tweeting any tweet. 

        :param retweet_histogram: 
        
        The bin and count values that denote the frequency of different retweet counts.

        :param epsilon: 
        
        Choose the highest activated result epsilon% of the time; (1-epsilon)% of the time use the probability distribution given to make a more random choice.  

        :param epochs: 
        
        How many times to give each user an opportunity to retweet something.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place all the "user" (those who retweet) vertices on the graph.
        4) Generate retweet_counts array (size m) based on `retweet_histogram`, element `i` in this array corresponds to the number of retweets user i will make.
        5) For each user in the graph, repeat retweet_counts[i] times:
            a) Calcualte the k dimensional probability vector,P1, based on the predfined neural network with the user's topic tweet history fed through.
            b) Of the possible edges, add one to the winning topic based on the calcualted probability distribution. 
                i) Update posteriors. 
    """
    assert epsilon <= 1 and epsilon >= 0, "Epsilon must be in the range [0..1]"
    # Add an extra element in the topics array that when selected the user doesn't retweet anything.
    if verbose and use_model: print("--- Loading Next Topic Neural Network ---")
    NEXT_TOPIC_NN = load_model("../neuralnet/dense_next_topic.h5") if use_model else None
    topics = [i for i in range(k)]
    G = init_graph(n,tweet_dist,k,m)
    # initialize the probability array
    pbar = tqdm.tqdm(total=m) if verbose else None
    retweets_per_user = sample_from_histogram(retweet_histogram,m) 
    for j,d in enumerate(retweets_per_user):
        if verbose: pbar.update(1)
        username = "user_{}".format(j)
        topic_history = G.nodes()[username]["topic_history"].copy()
        # Return the independent probabilities of choosing each topic, based on the agent's previous actions.
        for _ in range(d):
            topic_distribution = predict_next_retweet(topic_history, NEXT_TOPIC_NN, use_model=use_model)
            winning_topic = np.random.choice(topics, p=topic_distribution)
            if np.random.random() < epsilon and not np.all(topic_history == 0):
                arg_maxes = np.flatnonzero(
                    topic_distribution == topic_distribution.max())
                winning_topic = np.random.choice(arg_maxes)
            pos_tweets = possible_tweets(G, username, topic=winning_topic)
            if len(pos_tweets) > 0:
                winning_tweet = np.random.choice(pos_tweets)
                topic_history[winning_topic] += 1
                G.nodes[username]["topic_history"] = topic_history
                G.add_edge(winning_tweet, username)
    if verbose: pbar.close()
    return G

def stochastic_party_leader_graph(n=5, tweet_dist=(1000, 300), k=7, m=60000,retweet_histogram=None, epsilon=0.95,use_model=True,verbose=False,**kwargs):
    """
        Build a stochastic block model based o00 of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param n: 
        
        The number of "party leaders"

        :param tweet_dist: 
        
        A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: 
        
        The number of topics that a tweeter might tweet about.

        :param m: 
        
        The max number of retweeters in the system. 

        :param retweet_histogram: 
        
        The bin and count values that denote the frequency of different retweet counts.

        :param epsilon: 
        
        What % of the time you will choose the highest activated leader from the ANN.

        :param epochs: 
        
        How many times to give each user an opportunity to retweet something.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place all the "user" (those who retweet) vertices on the graph.
        4) Generate retweet_counts array (size m) based on `retweet_histogram`, element `i` in this array corresponds to the number of retweets user i will make.
        5) For each user in the graph, repeat retweet_counts[i] times:
            a) Calcualte the n dimensional probability vector,P, based on the predfined neural network with the user's leader tweet history fed through.
            b) Of the possible edges, add one to the winning leader based on the calcualted probability distribution. 
                i) Update posteriors. 
    """
    assert epsilon <= 1 and epsilon >= 0, "Epsilon must be in the range [0..1]"
    if verbose and use_model: print("--- Loading Next leader Neural Network ---")
    NEXT_LEADER_NN = load_model("../neuralnet/dense_next_leader.h5") if use_model else None
    # Add an extra element in the topics array that when selected the user doesn't retweet anything.
    topics = [i for i in range(k)]
    leaders = [i for i in range(n)]
    G = init_graph(n,tweet_dist,k,m)
    pbar = tqdm.tqdm(total=m) if verbose else None
    retweets_per_user = sample_from_histogram(retweet_histogram,m) 
    for j,d in enumerate(retweets_per_user):
        if verbose: pbar.update(1)
        username = "user_{}".format(j)
        leader_history = G.nodes()[username]["leader_history"].copy()
        for _ in range(d):
            # Return the independent probabilities of choosing each topic, based on the agent's previous actions.
            leader_distribution = predict_next_retweet(leader_history, NEXT_LEADER_NN, use_model=use_model)
            winning_leader = np.random.choice(leaders, p=leader_distribution)
            if np.random.random() < epsilon and not np.all(leader_history == 0):
                arg_maxes = np.flatnonzero(leader_distribution == leader_distribution.max())
                winning_leader = np.random.choice(arg_maxes)
            pos_tweets = possible_tweets(G, username, leader=winning_leader)
            if len(pos_tweets) > 0:
                winning_tweet = np.random.choice(pos_tweets)
                leader_history[winning_leader] += 1
                G.nodes[username]["leader_history"] = leader_history
                G.add_edge(winning_tweet, username)
    if verbose: pbar.close()
    return G

def stochastic_hybrid_graph(alpha=0.8,n=5, tweet_dist=(1000, 300), k=7, m=60000,retweet_histogram=None, epsilon=0.95,use_model=True,verbose=False,**kwargs):
    """
        Build a stochastic block model based off of the prior probability distributions of topic and leader engagment.
        Parameters
        ----------
        :param alpha: 
        
        The proportionate weightings of the leader selection and the topic selection. alpha -> 1: model is equivalent to stochastic_party_leader_graph; alpha -> 0 model is equivalent to stochastic_topic_graph

        :param n: 
        
        The number of "party leaders"

        :param tweet_dist: 
        
        A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: 
        
        The number of topics that a tweeter might tweet about.

        :param m: 
        
        The max number of retweeters in the system. 

        :param retweet_histogram: 
        
        The bin and count values that denote the frequency of different retweet counts.

        :param epsilon: 
        
        What % of the time you will choose the highest activated leader from the ANN.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place all the "user" (those who retweet) vertices on the graph.
        4) Generate retweet_counts array (size m) based on `retweet_histogram`, element `i` in this array corresponds to the number of retweets user i will make.
        5) For each user in the graph, repeat retweet_counts[i] times:
            a) Calcualte the k dimensional probability vector,P1, based on the predfined neural network with the user's topic tweet history fed through.
            b) Calcualte the n dimensional probability vector,P2, based on the predfined neural network with the user's leader tweet history fed through.
            c) Sum (P1.T*(1-alpha))+(P2*alpha) to get a kxn dimensional matrix
            d) Using these relative weightings, select an element (i,j) where i is the topic selected, and j is the party leader selected.
            c) Of the possible edges, add one to the winning leader/topic intersection. 
                i) Update posteriors. 
    """
    assert epsilon <= 1 and epsilon >= 0, "Epsilon must be in the range [0..1]"
    # Add an extra element in the topics array that when selected the user doesn't retweet anything.
    if verbose and use_model: print("--- Loading Next Topic Neural Network ---")
    NEXT_TOPIC_NN = load_model("../neuralnet/dense_next_topic.h5") if use_model else None
    if verbose and use_model: print("--- Loading Next Leader Neural Network ---")
    NEXT_LEADER_NN = load_model("../neuralnet/dense_next_leader.h5") if use_model else None
    topics = [i for i in range(k)]
    G = init_graph(n,tweet_dist,k,m)
    if verbose: print("--- Adding Retweets ---")
    pbar = tqdm.tqdm(total=m) if verbose else None
    retweets_per_user = sample_from_histogram(retweet_histogram,m) 
    topic_leader = [i for i in range(n*k)]
    for j,d in enumerate(retweets_per_user):    
        if verbose: pbar.update(1)
        username = "user_{}".format(j)
        topic_history = G.nodes()[username]["topic_history"].copy()
        leader_history = G.nodes()[username]["leader_history"].copy()
        for _ in range(d):
            # Return the independent probabilities of choosing each topic, based on the agent's previous actions.
            topic_distribution = predict_next_retweet(topic_history, NEXT_TOPIC_NN, use_model=use_model).reshape((1, k))
            leader_distribution = predict_next_retweet(leader_history, NEXT_LEADER_NN, use_model=use_model).reshape((1, n))
            topic_distribution *= (1-alpha)
            leader_distribution *= alpha
            # normalize 
            if k>n:
                topic_distribution *= (k/n)
            else:
                leader_distribution *= (n/k)
            # Add the weighted distributions together. Element (i,j) in this
            # matrix means the weight of choosing a tweet about topic i from party leader j
            topic_leader_matrix = topic_distribution.T+leader_distribution
            flattened_topic_leader = softmax(topic_leader_matrix.reshape(n*k))
            topic_leader_ind = np.random.choice(topic_leader, p=flattened_topic_leader)
            if np.random.random() < epsilon and not np.all(topic_history == 0) and not np.all(leader_history == 0):
                arg_maxes = np.flatnonzero(np.abs(flattened_topic_leader - flattened_topic_leader.max()) < 0.003)
                topic_leader_ind = np.random.choice(arg_maxes)
            winning_topic, winning_leader = np.unravel_index(topic_leader_ind, (k, n))
            pos_tweets = possible_tweets(G, username, leader=winning_leader, topic=winning_topic)
            if len(pos_tweets) > 0:
                # print(winning_leader,leader_distribution)
                winning_tweet = np.random.choice(pos_tweets)
                leader_history[winning_leader] += 1
                topic_history[winning_topic] += 1
                G.nodes()[username]["topic_history"] = topic_history
                G.nodes()[username]["leader_history"] = leader_history
                G.add_edge(winning_tweet, username)
    if verbose: pbar.close()
    return G


"""
Twitter Data
    * Num Tweets:           7978
    * Num Retweet Users:    36450
    * Num Total Retweets:   113293
"""
if __name__ == "__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    retweet_histogram = Graph(usernames).retweet_histogram()
    sample_g = Graph(usernames,n=50)
    sample_g.draw_graph(save=True)
    kwargs = {
        "tweet_dist": (sample_g.num_tweets, sample_g.num_tweets//5),
        "n": 5,
        "m": sample_g.num_retweeters,
        "epsilon": 0.9,
        "use_model": False,
        "verbose": True,
        "retweet_histogram": retweet_histogram
    } 
    str_args = str({i:kwargs[i] for i in kwargs if i!='retweet_histogram'})
    party_file_name="stochastic_party_leader_{}".format(str_args)
    party_G = stochastic_party_leader_graph(**kwargs)
    draw_graph(party_G,save=True,file_name=party_file_name,title="Party Leader Graph")
    topic_file_name="stochastic_topic_{}".format(str_args)
    topic_G = stochastic_topic_graph(**kwargs)
    draw_graph(topic_G,save=True,file_name=topic_file_name,title="Topic Graph")
    for alpha in np.round(np.arange(1,-0.01,-0.1),3).tolist():
        print("--- alpha {} --".format(alpha))
        hybrid_file_name = "stochastic_hybrid_graph_alpha={}_{}".format(alpha,str_args)
        hybrid_G = stochastic_hybrid_graph(alpha=alpha,**kwargs)
        draw_graph(hybrid_G, save=True, file_name=hybrid_file_name, title="Hybrid Graph. Alpha={}".format(alpha))
