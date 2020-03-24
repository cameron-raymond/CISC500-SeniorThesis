import sys
import json
import tqdm
import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl
import multiprocessing as mp
import matplotlib.pyplot as plt
from config import config
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

def plot_heat_traces(heat_dict,is_normalized=True,save_fig=False,benchmark=None,n=None,start=-2,end=2,file_name="heat_trace_plot"):
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
    min_time,max_time = np.min(np.logspace(start,end,250)), np.max(np.logspace(start,end,250))
    indices_to_keep = np.where((times >= min_time) & (times <=max_time))
    times = times[indices_to_keep]
    numeric_keys = np.array([nk for nk in heat_dict.keys() if __is_numeric(nk)])
    if len(numeric_keys):
        cmap = mpl.cm.bwr
        norm = mpl.colors.Normalize(vmin=np.min(numeric_keys), vmax=np.max(numeric_keys))
        axcb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        axcb.set_label('Alpha')
    for label,heat_traces in heat_dict.items():
        if type(heat_traces) is list:
            heat_traces = np.array(heat_traces)
            std_dev = np.array(heat_traces.std(axis=0)).flatten()[indices_to_keep]
            heat_traces = np.array(heat_traces.mean(axis=0)).flatten()[indices_to_keep]
            ax.fill_between(times, heat_traces+std_dev, heat_traces-std_dev, alpha=0.5)
        if __is_numeric(label):
            ax.plot(times,heat_traces,c=plt.cm.bwr(label))
        else:            
            ax.plot(times,heat_traces[indices_to_keep],label=label)
    ax.legend(loc="best")
    if benchmark:
        ax = fig.add_subplot(2,1,2)
        ax.set_title("Hybrid Model Comparison (Benchmark size={})".format(n))
        ax.set_xlabel("Alpha Value")
        ax.set_ylabel("Heat Trace Distance From Benchmark")
        distances = []
        t_alphas = []
        benchmark = heat_dict.pop(benchmark)[indices_to_keep]
        for alpha,heat_traces in sorted(heat_dict.items()):
            is_list = type(heat_traces) is list
            alphas = list(np.full(len(heat_traces),alpha)) if is_list else alpha
            heat_trace_dif = [compare(benchmark,ht[indices_to_keep]) for ht in heat_traces] if is_list else compare(benchmark,heat_traces[indices_to_keep])
            distances.append(heat_trace_dif)
            t_alphas.append(alpha)
            ax.plot(alphas,heat_trace_dif,'x',c="blue")
        distances,t_alphas = np.array(distances),np.array(t_alphas)
        std_dev = np.array(distances.std(axis=1)).flatten()
        distances = np.array(distances.mean(axis=1)).flatten()
        ax.fill_between(t_alphas, distances+std_dev, distances-std_dev, alpha=0.5,color='pink')
        ax.plot(t_alphas, distances,color='pink')
        ticks = np.round(ax.xaxis.get_majorticklocs(),3)
        zer_ind = np.where(ticks == 0)[0]
        one_ind = np.where(ticks == 1)[0]
        ticks = list(ticks)
        if zer_ind.size > 0:
            zer_ind = zer_ind[0]
            ticks[zer_ind] = "{} (Only Topic)".format(ticks[zer_ind])
        if one_ind.size > 0:
            one_ind = one_ind[0]
            ticks[one_ind] = "{} (Only Topic)".format(ticks[one_ind])
        ax.set_xticklabels(ticks)
    plt.tight_layout()
    file_name = "normalized_{}".format(file_name) if is_normalized else file_name
    file_name = "comparison_{}".format(file_name) if benchmark is not None else file_name
    plt.savefig("../visualizations/heat_traces/{}.png".format(file_name)) if save_fig else plt.show()

def dump_dict(a_dict,file_name="heat_traces"):
    with open("{}.json".format(file_name),'wb') as fp:
        pickle.dump(a_dict, fp)
        fp.flush()

def load_dict(file_name):
    try:
        with open("{}.json".format(file_name), 'rb') as fp:
            ret_dict = pickle.load(fp)
        return ret_dict
    except:
        return {}
        
if __name__ == "__main__":
    retweet_histogram = Graph(config["usernames"]).retweet_histogram()
    sample_g = Graph(config["usernames"],n=config["num_tweets"])
    graph_dict = {"Original Graph": sample_g.G}
    heat_dict_fn = "heat_traces_{}".format(str(config["kwargs"]))
    heat_dict = load_dict(heat_dict_fn)
    alphas = [float(sys.argv[1])] if sys.argv[1:] else config["alphas"]
    pbar = tqdm.tqdm(total=len(alphas)*config["num_per_alpha"])
    for alpha in alphas:
        gs = []
        if alpha not in heat_dict:
            for _ in range(config["num_per_alpha"]):
                gs.append(stochastic_hybrid_graph(alpha,m=sample_g.num_retweeters,retweet_histogram=retweet_histogram,**config["kwargs"]))
                pbar.update(1)
            graph_dict[alpha] = gs
            draw_graph(graph_dict[alpha][-1],save=config["save"],file_name="stochastic_hybrid_graph_alpha={:.3f}_{}".format(alpha,str(config["kwargs"])),title="Hybrid Graph. Alpha={}".format(alpha))
        else:
            pbar.update(config["num_per_alpha"])
            print("Skipping: {}".format(alpha))
    pbar.close()
    heat_dict.update(calc_heat(graph_dict=graph_dict))
    dump_dict(heat_dict,heat_dict_fn) if config["save"] else None
    plot_heat_traces(heat_dict,save_fig=config["save"],benchmark="Original Graph",n=config["num_tweets"],file_name=heat_dict_fn,start=0,end=1)
