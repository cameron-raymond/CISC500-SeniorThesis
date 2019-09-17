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
                colors.append('red')
            else:
                colors.append('green')
        else:
            colors.append('blue')

    plt.figure(figsize=(8, 8))
    # use graphviz to find radial layout
    pos = graphviz_layout(G, prog="twopi", root=0)
    # draw nodes, coloring by rtt ping time
    nx.draw(G, pos,
            node_color=colors,
            with_labels=False,
            alpha=0.5,
            node_size=15)
    # adjust the plot limits
    xmax = 1.02 * max(xx for xx, yy in pos.values())
    ymax = 1.02 * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.savefig("test.pdf", bbox_inches="tight")


if __name__ == '__main__':
    # Read in CSV file for that twitter user (these are the original tweets)
    username = sys.argv[1]
    twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
    # Instantiate a new MultiDiGraph (graph is directional + there could potentially be multip edges between a pair of nodes)
    G = nx.MultiDiGraph()
    nodes = twitter_df.set_index('id').to_dict('index').items()
    G.add_nodes_from(nodes)
    G.add_node("MaximeBernier")
    i = 0
    for node in G.nodes():
        G.add_edge("MaximeBernier", node)
        if i< 5:
            retweet_df = get_retweets(node).set_index('id').to_dict('index').items()
            G.add_nodes_from(retweet_df)
            for retweet in retweet_df:
                    G.add_edge(node, retweet[0])
            i+=1

    draw_graph(G)
