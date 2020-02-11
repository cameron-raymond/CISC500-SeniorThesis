import sys
import networkx as nx
import collections
import matplotlib.pyplot as plt
import numpy as np
from build_graph import Graph
from stochastic_block_model import stochastic_hybrid_graph
from sklearn.decomposition import PCA
import pandas as pd
# from scipy.spatial.distance import directed_hausdorff
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

def heat(G=None,graph_dict=None):
    assert G is not None or graph_dict is not None, "Need to supply a graph, or graphs, via the parameters G or graph_dict"
    to_heat = lambda eigs,t : np.sum(np.exp(-eigs*t))
    # Create a set of times ranging from 10**-4, to 10**2 on a logarithmic scale
    times = np.logspace(-4,2).tolist()
    heat_dict = {"t":times}
    if G is not None:
        print("Calculating eigenvals")
        eigenvals = nx.normalized_laplacian_spectrum(G)
        heats = np.array([to_heat(eigenvals,t) for t in times])
        heat_dict["graph"] = heats
    else:
        for label, graph in graph_dict.items():
            print("--- calculating {} eigenvalues (n={})".format(label,len(graph)))
            eigenvals = nx.normalized_laplacian_spectrum(graph)
            heats = np.array([to_heat(eigenvals,t) for t in times])
            heat_dict[label] = heats            
    return heat_dict

def plot_heat_traces(heat_dict,normalize=False):
    if normalize:
        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(2,1,1) 
    else:
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('t', fontsize = 15)
    ax.set_ylabel('h(t)', fontsize = 15)
    ax.set_xscale('log')
    ax.grid()
    times = heat_dict["t"]
    for label,heat_traces in heat_dict.items():
        if label != "t":
            ax.plot(times,heat_traces,label=label)
    ax.legend()
    if normalize:
        ax = fig.add_subplot(2,1,2)
        for label,heat_traces in heat_dict.items():
            if label != "t":
                heat_traces = (heat_traces-np.min(heat_traces))/(np.max(heat_traces)-np.min(heat_traces))
                ax.plot(times,heat_traces,label=label)
        ax.set_xlabel('t', fontsize = 15)
        ax.set_ylabel('h(t) (Normalized)', fontsize = 15)
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

    plt.show()


# I feel like this is gonna be important 
#L = nx.normalized_laplacian_matrix(G)
# and https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
if __name__=="__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    G = Graph(usernames,n=50).G  
    G2 = nx.erdos_renyi_graph(len(G),0.9)
    G3 = stochastic_hybrid_graph(alpha=0.8, tweet_dist=(50,10), n=5,m=100, tweet_threshold=0.37, epochs=9, epsilon=0.95,use_model=False)
    graph_dict = { "Actual Graph": G, "Erdos Renyi": G2,"Hybrid Graph (a=0.8)":G3}
    heat_dict = heat(graph_dict=graph_dict)
    plot_heat_traces(heat_dict)
    plot_heat_traces(heat_dict,normalize=True)

# if __name__ == "__main__":
#     m = 3
#     N = 100000
#     # usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
#     # G = Graph(usernames,n=100).G   
#     # graph_spectrum_laplacian(G,None)
#     tweet_dist = (50,0)
#     n       = 5
#     m       = 200
#     epochs  = 9
#     tweet_threshold = 0.37
#     epsilon = 0.95
#     frames = []
#     alphas = []
#     for i in range(2):
#         alphas += np.round(np.arange(1,-0.01,-0.1),3).tolist()
#     for alpha in alphas:
#         print("--- alpha: {} ---".format(alpha))
#         frames.append(graph_spectrum_laplacian(stochastic_hybrid_graph(alpha=alpha, tweet_dist=tweet_dist, n=n,m=m, tweet_threshold=tweet_threshold, epochs=epochs, epsilon=epsilon,use_model=False)))
#     frames = np.matrix(frames)
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(frames)
#     principal_df = pd.DataFrame(data=principalComponents,columns = ['pc1', 'pc2'])
#     principal_df = principal_df.assign(alpha=pd.Series(alphas).values)
#     principal_df.set_index('alpha', inplace=True)
#     fig = plt.figure(figsize = (8,8))
#     ax = fig.add_subplot(1,1,1) 
#     ax.set_xlabel('Principal Component 1', fontsize = 15)
#     ax.set_ylabel('Principal Component 2', fontsize = 15)
#     ax.set_title('2 component PCA', fontsize = 20)
#     principal_df.plot(kind='scatter',x='pc1',y='pc2',ax=ax)
#     for k, v in principal_df.iterrows():
#         ax.annotate("a: {}".format(k), v)
#     ax.grid()
#     plt.show()
    