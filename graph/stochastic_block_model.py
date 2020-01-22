import networkx as nx
import numpy as np
from numpy.random import normal,random
from build_graph import return_colour, return_legend
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference    

def stochastic_topic_graph(priors,n=5,tweet_dist=(1000,300),k=7,m=60000,p=0.02,alpha=0.9):
    """
        Build a stochastic block model based off of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param priors: A list of prior observations as to how retweeters have behaved. 

        :param n: The number of "party leaders"

        :param tweet_dist: A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: The number of topics that a tweeter might tweet about.

        :param m: The max number of retweeters in the system. 

        :param p: The default probability of a retweeter tweeting any tweet. 

        :param alpha: How heavily to weight the probability distributions.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place a "user" (those who retweet) vertice on the graph. Form an edge based off of the probability distribution, softmax(P) where P is a
            k dimensional vector where all elements are initialized to p). For the winning topic place an edge to one of the tweets of that topic randomly 
            (regardless of who tweeted it). 
        4) On each iteration, for each user in the graph:
            a) Calcualte the k dimensional probability vector,P, where P(topic i) = alpha*P(topic i | past topic retweet counts) for all i in k.
            b) Of the possible edges, add one ot the winning topic based on the calcualted probability distribution. 
                i) Update priors. 
        5) Repeat step 3-4 with a new user.
        6) Repeat until steps 3-5, until there are m retweeters have been placed on the graph or network has converged (no new edges added).
    """ 
    G = nx.Graph()
    for user in range(n):
        G.add_node(user, type='user')

    print("--- Adding party leader tweets ---")
    for user in range(n):
        num_tweets = max(int(normal(loc=tweet_dist[0],scale=tweet_dist[1])),100)
        topic_distribution = softmax([random() for i in range(k)])
        # here, a= are the topic numbers [0...6] and p is the probability of choosing tweet a
        tweets = np.random.choice(a=[i for i in range(k)],size=num_tweets,p=topic_distribution)
        pbar = tqdm.tqdm(total=num_tweets)
        for tweet,topic in enumerate(tweets):
            node = "{}_{}".format(user,tweet)
            G.add_node(node,type="tweet",lda_cluster=topic)
            G.add_edge(user,node)
            pbar.update(1)
        pbar.close()
    return G

def draw_graph(G, save=False, file_type='png'):
    """
    Handles rendering and drawing the network.
    Parameters
    ----------
    :param G: `optional` a networkx graph. If present draws this graph instead of the one built in the constructor.

    :param save: `optional` A boolean. If true saves an image of the graph to `/visualizations` otherwise renders the graph.
    
    :param file_type: `optional` A string. If save flag is true it saves graph with this file extension.
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
                colors.append('#79BFD3')
            elif attributes['type'] == 'tweet':
                cluster = return_colour(attributes["lda_cluster"])
                legend.add(cluster)
                colors.append(cluster[0])
            elif attributes['type'] == 'user':
                labels[node] = node
                colors.append('red')
        pbar.update(1)
    pbar.close()
    plt.figure(figsize=(30, 30))
    pos = graphviz_layout(G, prog="sfdp")
    print("--- Drawing {} nodes and {} edges ---".format(len(G.nodes()), G.number_of_edges()))
    nx.draw(G, pos,
            node_color=colors,
            with_labels=False,
            alpha=0.75,
            node_size=8,
            width=0.3,
            arrows=False
            )
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color='r')
    plt.legend(handles=return_legend(legend), loc="best")
    plt.show()

def stochastic_party_leader_graph(priors,n=5,tweet_dist=(1000,100),k=7,m=60000,p=0.02,alpha=0.9):
    """
        Build a stochastic block model based off of the prior probability distributions of topic engagment.
        Parameters
        ----------
        :param priors: A list of prior observations as to how retweeters have behaved. 

        :param n: The number of "tweeters"

        :param tweet_dist: A tuple where the first element is the mean number of tweets per tweeter, and the second element is the standard deviation of tweets.

        :param k: The number of topics that a tweeter might tweet about.

        :param m: The max number of retweeters in the system. 

        :param p: The default probability of a retweeter tweeting any tweet. 

        :param alpha: How heavily to weight the probability calculations distributions.

        Algorithm
        ---------
        1) Place n "party leader" (those who tweet) vertices on the graph. 
        2) For each of the n tweeters, assign x number of tweets to each (with some distribution of topics), based off of the tweet_dist parameter.
        3) Place a "user" (those who retweet) vertice on the graph. Form an edge based off of the probability distribution, softmax(P) where P is a
            n dimensional vector where all elements are initialized to p). For the winning leader, place an edge to one of their tweets.
            regardless of the topic.  
        4) On each iteration, for each user in the graph:
            a) Calcualte the n dimensional probability vector,P, where P(leader i) = alpha*P(leader i | past leader retweet counts) for all i in n.
            b) Of the possible edges, add one to the winning leader based on the calcualted probability distribution. 
                i) Update priors. 
        5) Repeat step 3-4 with a new user.
        6) Repeat until steps 3-5, until there are m retweeters have been placed on the graph or network has converged (no new edges added).
    """ 
    pass

if __name__ == "__main__":
    G = stochastic_topic_graph(None)
    draw_graph(G)