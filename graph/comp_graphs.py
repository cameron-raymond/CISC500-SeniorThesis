import sys
import networkx as nx
import collections
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def log_bin_frequency(G,title="Log-Log Histogram Plot",type=None,G2=None):
    degree_list=nx.degree(G)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degree_count = collections.Counter(degree_sequence)
    kmin=min(degree_count.values())
    kmax=max(degree_count.values())
    log_bins = np.logspace(np.log10(kmin), np.log10(kmax),num=20)
    log_bin_density, _ = np.histogram(degree_list, bins=log_bins, density=True)
    log_bins = np.delete(log_bins, -1)
    for x in range(len(log_bins)):
        log_bins[x] = ceil(log_bins[x])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title=(title)
    plt.xlabel("Log Degree")
    plt.xlabel("Log Frequency")
    plt.plot(log_bins,log_bin_density,'x',color='blue')
    plt.show()



if __name__ == "__main__":
    m = 3
    N = 900

    G = nx.barabasi_albert_graph(N, m)
    log_bin_frequency(G)

    