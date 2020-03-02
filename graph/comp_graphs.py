import sys
import json
import tqdm
import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from build_graph import Graph
from netlsd import heat, compare
from stochastic_block_model import stochastic_hybrid_graph, draw_graph

def calc_heat(G=None,graph_dict=None,start=-2,end=2,normalization="empty"):
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

def plot_heat_traces(heat_dict,is_normalized=True,save_fig=False,benchmark=None,n=None):
    assert benchmark is None and n is None or benchmark is not None and n is not None
    __is_numeric = lambda x : isinstance(x, (int, float, complex))
    heat_dict = heat_dict.copy()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(2,1,1) if benchmark else fig.add_subplot(1,1,1) 
    title = "Heat Trace (Normalized)" if is_normalized else "Heat Trace"
    ax.set_title(title)
    ax.set_ylabel('h(t)', fontsize = 15)
    ax.set_xscale('log')
    ax.grid()
    times = heat_dict.pop("t")
    numeric_keys = np.array([nk for nk in heat_dict.keys() if __is_numeric(nk)])
    if len(numeric_keys):
        cmap = mpl.cm.bwr
        norm = mpl.colors.Normalize(vmin=np.min(numeric_keys), vmax=np.max(numeric_keys))
        axcb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        axcb.set_label('Alpha')
    for label,heat_traces in heat_dict.items():
        if type(heat_traces) is list:
            heat_traces = np.array(heat_traces)
            std_dev = np.array(heat_traces.std(axis=0)).flatten()
            heat_traces = np.array(heat_traces.mean(axis=0)).flatten()
            ax.fill_between(times, heat_traces+std_dev, heat_traces-std_dev, alpha=0.5)
        if __is_numeric(label):
            ax.plot(times,heat_traces,c=plt.cm.bwr(label))
        else:            
            ax.plot(times,heat_traces,label=label)
    ax.legend(loc="best")
    if benchmark:
        ax = fig.add_subplot(2,1,2)
        ax.set_title("Hybrid Model Comparison (Benchmark n={})".format(n))
        ax.set_xlabel("Alpha Value")
        ax.set_ylabel("Heat Trace Distance From Benchmark")
        distances = []
        t_alphas = []
        for alpha,heat_traces in heat_dict.items():
            if alpha != benchmark:
                is_list = type(heat_traces) is list
                alphas = list(np.full(len(heat_traces),alpha)) if is_list else alpha
                heat_trace_dif = [compare(heat_dict[benchmark],ht) for ht in heat_traces] if is_list else compare(heat_dict[benchmark],heat_traces)
                distances.append(heat_trace_dif)
                t_alphas.append(alpha)
                ax.plot(alphas,heat_trace_dif,'x',c="blue")
        distances,t_alphas = np.array(distances),np.array(t_alphas)
        std_dev = np.array(distances.std(axis=1)).flatten()
        distances = np.array(distances.mean(axis=1)).flatten()
        ax.fill_between(t_alphas, distances+std_dev, distances-std_dev, alpha=0.5,color='pink')
        ax.plot(t_alphas, distances,color='pink')

    plt.tight_layout()
    file_name = '_'.join(map(str, heat_dict.keys())).replace(" ", "_").lower()
    file_name += "_heat_trace_plot"
    file_name = "normalized_"+file_name if is_normalized else file_name
    file_name = "comparison_"+file_name if benchmark is not None else file_name
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

def dump_dict(a_dict,file_name="heat_traces"):
    with open(file_name,'wb') as fp:
        pickle.dump(a_dict, fp)

def load_dict(file_name):
    with open(file_name, 'rb') as fp:
        ret_dict = pickle.load(fp)
    return ret_dict
        
if __name__ == "__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    retweet_histogram = Graph(usernames).retweet_histogram()
    n=2
    save = True
    num_per_alpha = 4
    sample_g = Graph(usernames,n=n)
    kwargs = {
        "tweet_dist": (n,n//5),
        "n": 5,
        "m": sample_g.num_retweeters,
        "epsilon": 0.9,
        "use_model": False,
        "verbose": False,
        "retweet_histogram": retweet_histogram
    } 
    str_args = str({i:kwargs[i] for i in kwargs if i!='retweet_histogram'})
    sample_g = sample_g.G
    graph_dict = {"Original Graph": sample_g}
    alpha_vals = np.round(np.arange(0,1.01,0.05),2)
    alpha_str = '_'.join(map(str, alpha_vals))
    pbar = tqdm.tqdm(total=len(alpha_vals)*num_per_alpha)
    for alpha in alpha_vals:
        gs = []
        for _ in range(num_per_alpha):
            gs.append(stochastic_hybrid_graph(alpha,**kwargs))
            pbar.update(1)
        graph_dict[alpha] = gs
        draw_graph(graph_dict[alpha][-1],save=save,file_name="stochastic_hybrid_graph_alpha={:.3f}_{}".format(alpha,str_args),title="Hybrid Graph. Alpha={}".format(alpha))
    pbar.close()
    heat_dict = calc_heat(graph_dict=graph_dict)
    dump_dict(heat_dict,"heat_traces_alphas={}_n={}_{}.json".format(alpha_str,n,str_args)) if save else None
    plot_heat_traces(heat_dict,save_fig=save,benchmark="Original Graph",n=n)

# if __name__ == "__main__":
#     usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
#     retweet_histogram = Graph(usernames).retweet_histogram()
#     n=215
#     og_graph = Graph(usernames,n=n)
#     kwargs = {
#         "tweet_dist": (n,n//5),
#         "n": 5,
#         "m": og_graph.num_retweeters,
#         "epsilon": 0.9,
#         "use_model": False,
#         "verbose": False,
#         "retweet_histogram": retweet_histogram
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