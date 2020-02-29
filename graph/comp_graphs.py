import sys
import networkx as nx
import collections
import matplotlib.pyplot as plt
import numpy as np
from build_graph import Graph
from stochastic_block_model import stochastic_hybrid_graph, draw_graph
from sklearn.decomposition import PCA
import pandas as pd
from netlsd import heat, compare
from math import ceil
import tqdm

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

def calc_heat(G=None,graph_dict=None,start=-3,end=2,normalization="empty"):
    assert G is not None or graph_dict is not None, "Need to supply a graph, or graphs, via the parameters G or graph_dict"
    # Create a set of times ranging from 10**-4, to 10**2 on a logarithmic scale
    times = np.logspace(start,end,250)
    heat_dict = {"t":times}
    if G is not None:
        heat_dict["graph"] = heat(G,timescales=times,normalization=normalization)
    else:
        for label, graph in graph_dict.items():
            if type(graph) is list:
                print("--- calculating heat traces for {} graphs of type {} ---".format(len(graph),label))
                heat_list = [heat(g,timescales=times, normalization=normalization) for g in graph]
                heat_dict[label] = heat_list
            else:
                print("--- calculating {} eigenvalues (n={})".format(label,len(graph)))
                heats = heat(graph,timescales=times,normalization=normalization)
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
            heat_traces = np.matrix(heat_traces)
            std_dev = np.array(heat_traces.std(axis=0)).flatten() * 2
            heat_traces = np.array(heat_traces.mean(axis=0)).flatten()
            ax.fill_between(times, heat_traces+std_dev, heat_traces-std_dev, alpha=0.5)
        ax.plot(times,heat_traces,label=label)
    ax.legend()
    file_name = "_".join(heat_dict.keys()).replace(" ", "_").lower()
    file_name += "_heat_trace_plot"
    file_name = "normalized_"+file_name if is_normalized else file_name
    plt.savefig("../visualizations/heat_traces/{}.png".format(file_name)) if save_fig else plt.show()

def fit_hybrid_model(target_graph,num_epochs=1000,learning_rate=0.01,min_delta=0.0001,**kwargs):
    """
        Fits a stochastic block model's alpha parameter to best model the heat trace of a target graph. 
        Parameters
        ----------
        :param target_graph: 
        
        A networkx graph that is to be modelled.

        :param num_epochs: 
        
        The number of epochs to attempt and fit the model to.

        :param learning_rate: 
        
        How much to scale the nudge to the alpha value on each iteration rate.

        :param min_delta: 
        
        The minimum change in the alpha value to be observed before exiting training.

        Algorithm
        ---------
        1) Initialize the hybrid model with some alpha value. 
        2) Calculate the difference in heat trace between that model and the target.
        3) Create a = learning_rate*difference and a2 = -learning_rate*difference.
            i) Recalculate heat trace difference, whichever one diminishes difference more make that the new alpha
            ii) Error = heat trace difference before - heat trace difference after
        3) On each epoch
            i) Calculate new model/heat trace difference with alpha
            ii) Calculate new error = difference before - difference after
            iii) alpha = alpha + learning_rate*error
        4) Repeat until epoch>=num_epochs or delta is sufficiently small.
    """
    size = 100
    target_trace = calc_heat(G=target_graph)["graph"]
    alpha = 0.8
    # alpha = np.random.uniform(0,1)
    history = []
    pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        hybrid_model = stochastic_hybrid_graph(alpha=alpha,**kwargs)
        hybrid_trace = calc_heat(G=hybrid_model)["graph"]
        heat_difference = compare(target_trace,hybrid_trace)
        a1,a2 = alpha + learning_rate*heat_difference, alpha - learning_rate*heat_difference
        a1_trace, a2_trace = calc_heat(nx.erdos_renyi_graph(size,a1))["graph"],calc_heat(nx.erdos_renyi_graph(size,a2))["graph"]
        e1, e2 = compare(target_trace,a1_trace),compare(target_trace,a2_trace)
        # a = alpha
        error = heat_difference
        if e1 < e2 and e1 < heat_difference:
            error = e1
            alpha = a1
        elif e2 < e1 and e2 < heat_difference:
            error = e2
            alpha = a2
        else:
            alpha = alpha + np.random.uniform(-0.01,0.01)*heat_difference
        history.append([epoch,alpha,error])
        # print("Epoch: {}".format(epoch))
        # if error < min_delta:
        #     return alpha
        # if a > alpha:
        #     print("Alpha from {:.10f} -> {:.10f}. Error: {}".format(a,alpha,error))
        pbar.update(1)
    pbar.close()
    return np.matrix(history)

# if __name__ == "__main__":
#     usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
#     og_graph = Graph(usernames=usernames,n=50)
#     kwargs = {
#         "tweet_dist": (50, 20),
#         "n": 5,
#         "m": 2686,
#         "epochs" : 2,
#         "tweet_threshold": 0.33,
#         "epsilon": 0.9,
#         "use_model": False,
#         "verbose": True
#     } 
#     print(og_graph.num_tweets,og_graph.num_retweeters,og_graph.num_retweets)
#     og_graph = og_graph.G
#     history = fit_hybrid_model(og_graph,num_epochs=5000,**kwargs)
#     epochs = history[:,0]
#     alphas = history[:,1]
#     errors = history[:,2]

#     print(epochs.shape,alphas.shape,errors.shape)
#     plt.subplot(3,1,1)
#     plt.title("Error/Alpha Breakdowns")
#     plt.plot(epochs,errors)
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.subplot(3,1,2)
#     plt.plot(epochs,alphas)
#     plt.xlabel('Epoch')
#     plt.ylabel('Alpha')
#     plt.subplot(3,1,3)
#     plt.ylabel('Error')
#     plt.xlabel('Alpha')
#     plt.plot(alphas,errors,'x')
#     plt.show()

    
if __name__ == "__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    retweet_histogram = Graph(usernames).retweet_histogram()
    sample_g = Graph(usernames,n=50)
    kwargs = {
        "tweet_dist": (sample_g.num_tweets, sample_g.num_tweets//5),
        "n": 5,
        "m": sample_g.num_retweeters,
        "epsilon": 0.9,
        "use_model": False,
        "verbose": False,
        "retweet_histogram": retweet_histogram
    } 
    graph_dict = {"Original Graph": sample_g}
    alpha_vals = np.round(np.arange(0,1.01,0.05),2)
    num_per_alpha = 3
    pbar = tqdm.tqdm(total=len(alpha_vals)*num_per_alpha)
    for alpha in alpha_vals:
        # print("--- generating hybrid (a={}) ---".format(alpha))
        graph_dict[alpha] = [stochastic_hybrid_graph(alpha,**kwargs) for _ in range(num_per_alpha)]
        pbar.update(num_per_alpha)
    pbar.close()
    heat_dict = calc_heat(graph_dict=graph_dict)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Hybrid Model Comparison (Benchmark n=10)")
    ax.set_xlabel("Alpha Value")
    ax.set_ylabel("Heat Trace Distance From Benchmark")
    for alpha,heat_traces in heat_dict.items():
        if alpha != "t" and alpha != "Original Graph":
            alphas = [alpha for _ in range(len(heat_traces))] if type(heat_traces) is list else alpha
            heat_trace_dif = [compare(heat_dict["Original Graph"],heat_trace) for heat_trace in heat_traces] if type(heat_traces) is list else compare(heat_dict["Original Graph"],heat_traces)
            ax.plot(alphas,heat_trace_dif,'x',c="blue")
    plt.savefig("../visualizations/heat_traces/hybrid_heat_trace_difference_a={}.png".format(str(alpha_vals)))
