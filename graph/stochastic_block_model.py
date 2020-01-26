import networkx as nx
import numpy as np
from numpy.random import normal, random
from tensorflow.keras.models import load_model
from build_graph import return_colour, return_legend
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm
from centrality_measures import centrality_per_topic, plot_dual_centralities

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def possible_tweets(G, aUser, topic=None,leader=None):
    """
        Returns all tweet nodes of a certain topic that have not already been retweeted by a user.
    """
    assert (topic is None or leader is None) and not (topic is None and leader is None), "Only one of the topic and leader can be present"
    possible_tweets = []
    for node in G.nodes():
        attributes = G.nodes[node]
        already_connected = node in G[aUser]
        if already_connected:
            pass
        if not topic is None:
            if attributes["type"] == "tweet" and attributes["lda_cluster"] == topic and not already_connected:
                possible_tweets.append(node)
        elif not leader is None:
            if attributes["type"] == "tweet" and leader in G[node] and not already_connected:
                possible_tweets.append(node)
    return possible_tweets


def predict_next_retweet(history,model,use_model=True):
    next_decision_distribution = history + np.full(history.shape,1)
    if use_model:
        history = history.reshape(1,len(history)) 
        next_decision_distribution = model.predict(history).reshape(history.shape[1])
    # Squish it using softmax to make it a probability distribution.
    return softmax(next_decision_distribution)
 

def stochastic_topic_graph(n=5, tweet_dist=(1000, 300), k=7, m=60000, tweet_threshold=0.5, epsilon=0.95,epochs=2.5):
    """
        Build a stochastic block model based off of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param posteriors: A list of prior observations as to how retweeters have behaved. 

        :param n: The number of "party leaders"

        :param tweet_dist: A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: The number of topics that a tweeter might tweet about.

        :param m: The max number of retweeters in the system. 

        :param p: The default probability of a retweeter tweeting any tweet. 

        :param tweet_threshold: A float on in the range [0..1]. A user will retweet any tweet this percent of the time.

        :param epsilon: Choose the highest activated result epsilon% of the time; (1-epsilon)% of the time use the probability distribution given to make a more random choice.  

        :param epochs: How many times to give each user an opportunity to retweet something.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place a "user" (those who retweet) vertice on the graph. Form an edge based off of the probability distribution, softmax(P) where P is a
            k dimensional vector where all elements are initialized to p). For the winning topic place an edge to one of the tweets of that topic randomly 
            (regardless of who tweeted it). 
        4) On each iteration, for each user in the graph:
            a) Calcualte the k dimensional probability vector,P, where P(topic i) = epsilon*P(topic i | past topic retweet counts) for all i in k.
            b) Of the possible edges, add one ot the winning topic based on the calcualted probability distribution. 
                i) Update posteriors. 
        5) Repeat step 3-4 with a new user.
        6) Repeat until steps 3-5, until there are m retweeters have been placed on the graph or network has converged (no new edges added).
    """
    assert tweet_threshold <= 1 and tweet_threshold >= 0, "Tweet threshold must be in the range [0..1]"
    assert epsilon <= 1 and epsilon >= 0, "Epsilon must be in the range [0..1]"    
    # Add an extra element in the topics array that when selected the user doesn't retweet anything.
    print("--- Loading Next Topic Neural Network ---")
    NEXT_TOPIC_NN = load_model("../neuralnet/dense_next_topic.h5")
    topics = [i for i in range(k)]
    G = nx.Graph()
    for user in range(n):
        G.add_node(user, type='user')

    print("--- Adding party leader tweets ---")
    for user in range(n):
        num_tweets = max(int(normal(loc=tweet_dist[0], scale=tweet_dist[1])), 1)
        topic_distribution = softmax([random() for i in range(k)])
        # here, a= are the topic numbers [0...6] and p is the probability of choosing tweet a
        tweets = np.random.choice(a=topics, size=num_tweets, p=topic_distribution)
        pbar = tqdm.tqdm(total=num_tweets)
        for tweet, topic in enumerate(tweets):
            node = "{}_{}".format(user, tweet)
            G.add_node(node, type="tweet", lda_cluster=topic)
            G.add_edge(user, node)
            pbar.update(1)
        pbar.close()

    history = np.zeros(k)
    print("--- adding tweets ---")
    #progress bar
    for new_user in range(m):
        # add a new user
        username = "user_{}".format(new_user)
        G.add_node(username, type="retweet", history=history)
    # initialize the probability array
    pbar = tqdm.tqdm(total=m*epochs)
    for _ in range(epochs):
        for j in range(m):
            pbar.update(1)
            username = "user_{}".format(j)
            history = G.nodes()[username]["history"].copy()
            # Return the independent probabilities of choosing each topic, based on the agent's previous actions.
            topic_distribution = predict_next_retweet(history,NEXT_TOPIC_NN,use_model=True)
            winning_topic = np.argmax(topic_distribution) if np.random.random() < epsilon and not np.all(history==0)  else np.random.choice(topics, p=topic_distribution)
            odds = np.random.random()
            pos_tweets = possible_tweets(G, username, topic=winning_topic)
            if len(pos_tweets) > 0 and odds > tweet_threshold:
                winning_tweet = np.random.choice(pos_tweets)
                history[winning_topic] += 1
                G.nodes[username]["history"] = history
                G.add_edge(winning_tweet, username)
    pbar.close()
    return G

def draw_graph(G, save=False, title="stochastic_block_graph",file_type='png',transparent=False):
    """
    Handles rendering and drawing the network.
    Parameters
    ----------
    :param G: `optional` a networkx graph. If present draws this graph instead of the one built in the constructor.

    :param save: `optional` A boolean. If true saves an image of the graph to `/visualizations` otherwise renders the graph.

    :param file_type: `optional` A string. If save flag is true it saves graph with this file extension.

    :param transparent: `optional` A Boolean. If true it only renders the tweet's; no edges or user nodes.
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
                colors.append('#79bfd3ff') if not transparent else "white"
            elif attributes['type'] == 'tweet':
                cluster = return_colour(attributes["lda_cluster"])
                legend.add(cluster)
                colors.append(cluster[0])
            elif attributes['type'] == 'user':
                labels[node] = node
                colors.append('red') if not transparent else "white"
        pbar.update(1)
    pbar.close()
    plt.figure(figsize=(30, 30))
    pos = graphviz_layout(G, prog="sfdp")
    print("--- Drawing {} nodes and {} edges ---".format(len(G.nodes()),
                                                         G.number_of_edges()))
    width = 0.3 if not transparent else o
    nx.draw(G, pos,
            node_color=colors,
            with_labels=False,
            alpha=0.75,
            node_size=8,
            width=width,
            arrows=False
            )
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color='r')
    plt.legend(handles=return_legend(legend), loc="best")
    plt.savefig("../visualizations/random_graphs/{}.{}".format(title,file_type)) if save else plt.show()

def stochastic_party_leader_graph(n=5, tweet_dist=(1000, 300), k=7, m=60000, tweet_threshold=0.5, epsilon=0.9,epochs=2.5):
    """
        Build a stochastic block model based o00 of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param n: The number of "party leaders"

        :param tweet_dist: A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: The number of topics that a tweeter might tweet about.

        :param m: The max number of retweeters in the system. 

        :param tweet_threshold: A float on in the range [0..1]. A user will retweet any tweet this percent of the time.

        :param epsilon: What % of the time you will choose the highest activated leader from the ANN.

        :param epochs: How many times to give each user an opportunity to retweet something.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place a "user" (those who retweet) vertice on the graph. Form an edge based off of the probability distribution, softmax(P) where P is a
            k dimensional vector where all elements are initialized to p). For the winning topic place an edge to one of the tweets of that topic randomly 
            (regardless of who tweeted it). 
        4) On each iteration, for each user in the graph:
            a) Calcualte the k dimensional probability vector,P, based on the predfined neural network with the users tweet history fed through.
            b) Of the possible edges, add one ot the winning leader based on the calcualted probability distribution. 
                i) Update posteriors. 
        5) Repeat step 3-4 with a new user.
        6) Repeat until steps 3-5, until there are m retweeters have been placed on the graph or network has converged (no new edges added).
    """
    assert tweet_threshold <= 1 and tweet_threshold >= 0, "Tweet threshold must be in the range [0..1]"
    assert epsilon <= 1 and epsilon >= 0, "Epsilon must be in the range [0..1]"
    print("--- Loading Next leader Neural Network ---")
    NEXT_LEADER_NN = load_model("../neuralnet/dense_next_leader.h5")
    # Add an extra element in the topics array that when selected the user doesn't retweet anything.
    topics = [i for i in range(k)]
    leaders = [i for i in range(n)]
    G = nx.Graph()
    for user in leaders:
        G.add_node(user, type='user')

    print("--- Adding party leader tweets ---")
    for user in range(n):
        num_tweets = max(int(normal(loc=tweet_dist[0], scale=tweet_dist[1])), 1)
        leader_distribution = softmax([random() for i in range(k)])
        # here, a= are the topic numbers [0...6] and p is the probability of choosing tweet a
        tweets = np.random.choice(a=topics, size=num_tweets, p=leader_distribution)
        pbar = tqdm.tqdm(total=num_tweets)
        for tweet, topic in enumerate(tweets):
            node = "{}_{}".format(user, tweet)
            G.add_node(node, type="tweet", lda_cluster=topic)
            G.add_edge(user, node)
            pbar.update(1)
        pbar.close()
    # Create an array where each element
    history = np.zeros(n)
    print("--- adding tweets ---")
    #progress bar
    for new_user in range(m):
        # add a new user
        username = "user_{}".format(new_user)
        G.add_node(username, type="retweet", history=history)
        # initialize the probability array
    pbar = tqdm.tqdm(total=m*epochs)
    for epoch in range(epochs):
        for j in range(m):
            pbar.update(1)
            username = "user_{}".format(j)
            history = G.nodes()[username]["history"].copy()
            # Return the independent probabilities of choosing each topic, based on the agent's previous actions.
            leader_distribution = predict_next_retweet(history,NEXT_LEADER_NN,use_model=True)
            winning_leader = np.argmax(leader_distribution) if np.random.random() < epsilon and not np.all(history==0)  else np.random.choice(leaders, p=leader_distribution)
            pos_tweets = possible_tweets(G, username, leader=winning_leader)
            odds = np.random.random()
            if len(pos_tweets) > 0 and odds > tweet_threshold:
                # print(winning_leader,leader_distribution)
                winning_tweet = np.random.choice(pos_tweets)
                history[winning_leader] += 1
                G.nodes[username]["history"] = history
                G.add_edge(winning_tweet, username)
    pbar.close()
    return G

def stochastic_hybrid_graph():
    pass


"""
Twitter Data
    * Num Tweets:           4416
    * Num Retweet Users:    23754
    * Num Total Retweets:   65897

Retweets Per Retweeter:  2.8    (epochs*(1-tweet_threshold)=2.8)
Retweeters Per Tweet: 5.37        (m/tweet_dist = 5.37)
"""
if __name__ == "__main__":
    tweet_dist = (200,0)
    n=5
    # m=447
    # epochs=10
    # tweet_threshold=0.3
    # epsilon=0.99
    m=1074
    epochs=20
    tweet_threshold=0.8
    epsilon=0.90
    party_title="stochastic_party_leader_tweet_dist={}_m={}_epochs={}_tweet_threshold={}".format(tweet_dist,m,epochs,tweet_threshold)
    party_G = stochastic_party_leader_graph(tweet_dist=tweet_dist,n=n, m=m,tweet_threshold=tweet_threshold,epochs=epochs,epsilon=epsilon)
    draw_graph(party_G,save=True,title=party_title)
    party_title="transparent_stochastic_party_leader_tweet_dist={}_m={}_epochs={}_tweet_threshold={}".format(tweet_dist,m,epochs,tweet_threshold)
    draw_graph(party_G,save=True,title=party_title)
    # topic_G = stochastic_topic_graph(tweet_dist=tweet_dist,n=n, m=m,tweet_threshold=tweet_threshold,epochs=epochs,epsilon=epsilon)
    # topic_title="stochastic_topic_tweet_dist={}_m={}_epsilon={}_epochs={}_tweet_threshold={}".format(tweet_dist,m,epsilon,epochs,tweet_threshold)
    # draw_graph(topic_G,save=True,title=topic_title)
    # zscore_overall_topic_centralities = centrality_per_topic(G, measure='zscore')
    # print(zscore_overall_topic_centralities)