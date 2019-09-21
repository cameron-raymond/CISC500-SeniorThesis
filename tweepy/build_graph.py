#!/usr/bin/python
import sys
import operator
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
from get_retweets import get_retweets
from tweet_config import COLS


def draw_graph(G):
    colors = []
    for node in G.nodes():
        attributes = G.node[node]
        if 'type' in attributes:
            if G.node[node]['type'] == 'retweet':
                colors.append('green')
            else:
                colors.append('red')
        else:
            colors.append('blue')

    plt.figure(figsize=(12, 12))
    # use graphviz to find radial layout
    # pos = graphviz_layout(G, prog="dot", root=0)
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    # draw nodes, coloring by rtt ping time
    nx.draw(G, pos,
            node_color=colors,
            with_labels=False,
            alpha=0.5,
            k=4,
            node_size=15)
    # adjust the plot limits
    xmax = 1.02 * max(xx for xx, yy in pos.values())
    ymax = 1.02 * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()
    # plt.savefig("test.pdf", bbox_inches="tight")


if __name__ == '__main__':
    # Read in CSV file for that twitter user (these are the original tweets)
    username = sys.argv[1]
    twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
    # Instantiate a new MultiDiGraph (graph is directional + there could potentially be multip edges between a pair of nodes)
    G = nx.MultiDiGraph()
    nodes = twitter_df.set_index('id').to_dict('index').items()
    G.add_nodes_from(nodes)
    G.add_node("MaximeBernier")
    for node in G.nodes()[:-1]:
        G.add_edge("MaximeBernier", node)
        retweet_df = get_retweets(node)
        user_nodes = retweet_df.drop_duplicates(subset ="original_author") 
        user_nodes = user_nodes.set_index('original_author').to_dict('index').items()
        G.add_nodes_from(user_nodes)
        for index, row in retweet_df.iterrows():
            G.add_edge(node, row['original_author'])

    draw_graph(G)
