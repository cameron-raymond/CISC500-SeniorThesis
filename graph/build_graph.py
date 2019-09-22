#!/usr/bin/python
import sys
import operator
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd


def draw_graph(G):
    print("--- Adding colours and labels ---")
    colors = []
    labels = {}
    retweet_labels = {}
    for node in G.nodes():
        attributes = G.node[node]
        if 'type' in attributes:
            if G.node[node]['type'] == 'retweet':
                retweet_labels[node] = node
                colors.append('#79BFD3')
            elif G.node[node]['type'] == 'user':
                labels[node] = node
                colors.append('red')
            else:
                colors.append('blue')
       
    print("--- Laying out nodes ---")
    plt.figure(figsize=(30, 30))
    # use graphviz to find radial layout
    pos = graphviz_layout(G, prog="twopi", root=0)
    # pos = nx.spring_layout(G, k=0.15, iterations=20)
    # draw nodes, coloring by rtt ping time
    nx.draw(G, pos,
            node_color=colors,
            with_labels=False,
            alpha=0.5,
            node_size=80,
            arrowsize=10,
            arrowstyle='fancy')
    nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='b')
    # nx.draw_networkx_labels(G,pos,retweet_labels,font_size=8,font_color='g')
    plt.savefig("test2.png", bbox_inches="tight")
    # plt.show()


if __name__ == '__main__' :
    # Read in CSV file for that twitter user (these are the original tweets)
    username = sys.argv[1]
    twitter_df = pd.read_csv("../data/{}_data.csv".format(username)).head(20)
    retweet_df = pd.read_csv("../data/{}_retweets.csv".format(username))
    retweet_df = retweet_df[retweet_df['original_tweet_id'].isin(twitter_df['id'])]
    # di = {'tweet': "red", 'retweet': "green"}
    # twitter_df.replace({"type": di})
    # twitter_df.rename(columns={'type':'population'})
    # Instantiate a new MultiDiGraph (graph is directional + there could potentially be multip edges between a pair of nodes)
    G = nx.MultiDiGraph()
    G.add_node(username,type='user')
    # add tweet nodes
    nodes = twitter_df.set_index('id').to_dict('index').items()
    G.add_nodes_from(nodes)
    for index,row in twitter_df.iterrows():
        G.add_edge(username,row['id'])
    # add retweet user nodes (those who retweeted the original tweets) multipl
    user_nodes = retweet_df.drop_duplicates(subset ="original_author") 
    user_nodes = user_nodes.set_index('original_author').to_dict('index').items()
    G.add_nodes_from(user_nodes)
    for index,row in retweet_df.iterrows():
        G.add_edge(row['original_tweet_id'],row['original_author'])
    
    draw_graph(G)
