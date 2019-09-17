#!/usr/bin/python
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tweet_config import COLS

if __name__ == '__main__':
    username = sys.argv[1]
    twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
    nodes = twitter_df.set_index('id').to_dict('index').items()
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    
    print(G.nodes(data=True)[0])

    # plt.figure(num=None, figsize=(20, 20), dpi=80)
    # plt.axis('off')
    # fig = plt.figure(1)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G,pos)
    # nx.draw_networkx_edges(G,pos)
    # nx.draw_networkx_labels(G,pos)

    # cut = 1.00
    # xmax = cut * max(xx for xx, yy in pos.values())
    # ymax = cut * max(yy for xx, yy in pos.values())
    # plt.xlim(0, xmax)
    # plt.ylim(0, ymax)

    # plt.savefig("test.pdf",bbox_inches="tight")
    # pylab.close()
    # del fig
   