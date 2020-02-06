import sys
import networkx as nx
import collections
import matplotlib.pyplot as plt
import numpy as np
from build_graph import Graph
from stochastic_block_model import stochastic_hybrid_graph
from scipy.spatial.distance import directed_hausdorff
# import Graph from build_graph
from math import ceil

def log_bin_frequency(G):
    # degree_list=list(nx.degree(G))
    degree_list = [d for n, d in G.degree()]
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degree_count = collections.Counter(degree_sequence)
    # print(list(degree_list))
    kmin=min(degree_count.values())
    kmax=max(degree_count.values())
    log_bins = np.logspace(np.log10(kmin), np.log10(kmax),num=20)
    log_bin_density, _ = np.histogram(degree_list, bins=log_bins, density=True)
    log_bins = np.delete(log_bins, -1)
    for x in range(len(log_bins)):
        log_bins[x] = ceil(log_bins[x])
    return log_bins, log_bin_density


def plot_log_bin_frequency(G,G2=None,title="Log-Log Histogram Plot",type=None):
    log_bins, log_bin_density = log_bin_frequency(G)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title=(title)
    plt.xlabel("Log Degree")
    plt.ylabel("Log Frequency")
    plt.plot(log_bins,log_bin_density,'x',color='blue',label="Original Graph")
    if G2 is not None:
        log_bins,log_bin_density = log_bin_frequency(G2)
        plt.plot(log_bins,log_bin_density,'x',color='pink',label="Model Graph")
    plt.legend(loc="best")
    plt.show()


# I feel like this is gonna be important 
#L = nx.normalized_laplacian_matrix(G)
# and https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html

if __name__ == "__main__":
    m = 3
    N = 100000
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    G = Graph(usernames).G   
    tweet_dist = (100, 35)
    n       = 5
    m       = 476
    epochs  = 9
    tweet_threshold = 0.37
    epsilon = 0.95
    alpha =0.8
    G2  = stochastic_hybrid_graph(alpha=alpha, tweet_dist=tweet_dist, n=n,m=m, tweet_threshold=tweet_threshold, epochs=epochs, epsilon=epsilon)
    # plot_log_bin_frequency(G,G2=G2)

    