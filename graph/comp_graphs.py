import sys
import networkx as nx
import collections
import matplotlib.pyplot as plt
import numpy as np
from build_graph import Graph
from stochastic_block_model import stochastic_hybrid_graph, draw_graph
from sklearn.decomposition import PCA
import pandas as pd
# from scipy.spatial.distance import directed_hausdorff
from math import ceil

def log_bin_frequency(G):
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

def calc_heat(graph,times,normalize):
    to_heat = lambda eigs,t : np.sum(np.exp(-eigs*t))
    eigenvals = nx.normalized_laplacian_spectrum(graph)
    heats = np.array([to_heat(eigenvals,t) for t in times])
    if normalize:
        # Eigenvalues of an empty graph are all 0's. Therefore e^(-t*0) is always 1. Since any graph has n 
        # eigenvalues the sum of the heat trace is n regardless of the timeframe
        heats = heats/len(graph)
    return heats

def heat(G=None,graph_dict=None,start=-3,end=2,normalize=True):
    assert G is not None or graph_dict is not None, "Need to supply a graph, or graphs, via the parameters G or graph_dict"
    # Create a set of times ranging from 10**-4, to 10**2 on a logarithmic scale
    times = np.logspace(start,end).tolist()
    heat_dict = {"t":times}
    if G is not None:
        heat_dict["graph"] = calc_heat(G,times,normalize)
    else:
        for label, graph in graph_dict.items():
            if type(graph) is list:
                print("--- calculating heat traces for {} graphs of type {} ---".format(len(graph),label))
                heat_list = [calc_heat(g,times,normalize) for g in graph]
                heat_dict[label] = heat_list
            else:
                print("--- calculating {} eigenvalues (n={})".format(label,len(graph)))
                heats = calc_heat(graph,times,normalize)
                heat_dict[label] = heats            
    return heat_dict

def plot_heat_traces(heat_dict,is_normalized=True,save_fig=False):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)  
    title = "Heat Trace (Normalized)" if is_normalized else "Heat Trace"
    ax.set_title(title)
    ax.set_ylabel('h(t)', fontsize = 15)
    ax.set_xscale('log')
    ax.grid()
    times = heat_dict.pop("t")
    for label,heat_traces in heat_dict.items():
        if type(heat_traces) is list:
            print("here")
            heat_traces = np.matrix(heat_traces)
            print(heat_traces)
            std_dev = np.array(heat_traces.std(axis=0)).flatten()
            print("///")
            print(std_dev)
            heat_traces = np.array(heat_traces.mean(axis=0)).flatten()
            print("///")
            print(heat_traces)
            ax.fill_between(times, heat_traces+std_dev, heat_traces-std_dev, alpha=0.5)
        ax.plot(times,heat_traces,label=label)
    ax.legend()
    file_name = "_".join(heat_dict.keys()).replace(" ", "_").lower()
    file_name += "_heat_trace_plot"
    file_name = "normalized_"+file_name if is_normalized else file_name
    plt.savefig("../visualizations/heat_traces/{}.png".format(file_name)) if save_fig else plt.show()

if __name__=="__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    sampled_graphs = [Graph(usernames,n=15).G for _ in range (10)]
    avg_size = int(np.mean([len(G) for G in sampled_graphs]))
    graph_dict = {"Original Graph": sampled_graphs}
    kwargs = {
        "tweet_dist": (100, 20),
        "n": 5,
        "m": 407,
        "epochs" : 9,
        "tweet_threshold": 0.37,
        "epsilon": 0.9,
        "use_model": False
    }
    normalize = True
    # graph_dict["Hybrid Graph"] = [stochastic_hybrid_graph(alpha=alpha,**kwargs) for alpha in np.round(np.arange(1,-0.01,-0.2),2)]
    graph_dict["Erdos Renyi"] = [nx.erdos_renyi_graph(avg_size,alpha) for alpha in np.round(np.arange(1,0,-0.2),2)]
    # for alpha in np.round(np.arange(1,-0.01,-0.5),2):
    #     print("a is {}".format(alpha))
    #     graph_dict["Hybrid Graph (a={})".format(alpha)] = [stochastic_hybrid_graph(alpha=alpha,**kwargs) for _ in range(10)]
    #     graph_dict["Erdos Renyi (p = {})".format(alpha)] = [nx.erdos_renyi_graph(avg_size,alpha) for _ in range(10)]
    heat_dict = heat(graph_dict=graph_dict,normalize=normalize)
    plot_heat_traces(heat_dict,is_normalized=normalize,save_fig=True)
    

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
    # frames = np.matrix(frames)
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(frames)
    # principal_df = pd.DataFrame(data=principalComponents,columns = ['pc1', 'pc2'])
    # principal_df = principal_df.assign(alpha=pd.Series(alphas).values)
    # principal_df.set_index('alpha', inplace=True)
    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel('Principal Component 1', fontsize = 15)
    # ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_title('2 component PCA', fontsize = 20)
    # principal_df.plot(kind='scatter',x='pc1',y='pc2',ax=ax)
    # for k, v in principal_df.iterrows():
    #     ax.annotate("a: {}".format(k), v)
    # ax.grid()
    # plt.show()
    