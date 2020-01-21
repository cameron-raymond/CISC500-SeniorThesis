import sys
from build_graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

NUM_TOPICS = 7
COLOURS = ["#006816","#8d34e4","#c9a738","#0163d0","#ee5700", "#00937e", "#ff4284", "#4b5400", "#ea80ff","#9f0040"]


def centrality_per_topic(G,username=None,mean=False):
    """
        Calculates the eigenvector centrality for a network G, and then takes the mean centrality of tweets grouped by topic. 
        Parameters
        ----------
        :param G: The networkx Graph to calculate centrality from
        
        :param username: `optional` if present only uses that user's tweets when calculating the mean.
                
        :param mean: `optional` If True calculates the average centrality for topic. Otherwise sums it up.
    """
    netx_graph = G.G.copy()
    # centrality = nx.eigenvector_centrality_numpy(netx_graph)
    centrality = nx.eigenvector_centrality_numpy(netx_graph)
    topic_centrality = dict((topic,[]) for topic in range(NUM_TOPICS))
    for node in netx_graph.nodes():
        node_attributes = netx_graph.nodes[node]
        node_centrality = centrality[node]
        if "lda_cluster" in node_attributes:
            if username and username in netx_graph.neighbors(node) or username is None:
                topic = node_attributes["lda_cluster"]
                if topic in topic_centrality:
                    topic_centrality[topic].append(node_centrality)
                else:
                    topic_centrality[topic] = [node_centrality]

    for topic,centrality_list in topic_centrality.items():
        if mean:
            topic_centrality[topic] = np.mean(centrality_list)
        else:
            topic_centrality[topic] = np.sum(centrality_list)
    return topic_centrality


def plot_dual_centralities(overall_net_cent,individaul_cents,expand=False,usernames=None,mean=False):
    """
        Plots the 

    """
    overall_net_cent = sorted(overall_net_cent.items(),key=lambda tup: tup[0])
    overall_net_cent = overall_net_cent[1:] if overall_net_cent[0][0] == -1 else overall_net_cent
    leader_cents = []
    for leader_cent in individaul_cents:
        leader_cent = sorted(leader_cent.items(),key=lambda tup: tup[0])
        leader_cent = leader_cent[1:] if leader_cent[0][0] == -1 else leader_cent # get rid of topic -1 (empty tweet)
        topics, centralities = map(list, zip(*leader_cent))
        centralities = np.nan_to_num(np.array(centralities))
        leader_cents.append(centralities)
    topics, average_overall_cents = map(list, zip(*overall_net_cent))
    label = "Average" if mean else "Total"
    plt.title('Overall Network Centrality/Individual\'s Network Centrality ({})'.format(label))
    plt.xlabel('{} Topic Centrality for Individual\'s Network'.format(label))
    plt.ylabel('{} Topic Centrality'.format(label))
    plt.grid(True)
    if expand and usernames:
        assert len(usernames) == len(leader_cents)
        for i,(username,centralities) in enumerate(zip(usernames,leader_cents)):
            plt.scatter(centralities, average_overall_cents, alpha=0.5,label=username,c=COLOURS[i])
            for j, txt in enumerate(topics):
                plt.annotate("{}".format(txt+1), (centralities[j], average_overall_cents[j])) 
            plt.legend()
        plt.savefig("../visualizations/centrality_charts/{}_opposing_centrality_chart_(expanded).png".format(label))
        plt.clf()
        return
    else:
        average_individual_cents = np.mean(leader_cents, axis=0)
        plt.scatter(average_individual_cents, average_overall_cents, alpha=0.5)
        for j, txt in enumerate(topics):
            plt.annotate("Topic {}".format(txt+1), (average_individual_cents[j], average_overall_cents[j]))
        plt.savefig("../visualizations/centrality_charts/{}_opposing_centrality_chart_(mean_of_leaders).png".format(label))
        plt.clf()
        return
if __name__ == "__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else ["JustinTrudeau","ElizabethMay","theJagmeetSingh","AndrewScheer","MaximeBernier"]
    G = Graph(usernames,n=10)
    sum_overall_topic_centralities = centrality_per_topic(G,mean=False)
    mean_overall_topic_centralities = centrality_per_topic(G,mean=True)
    sum_leader_cents = []
    mean_leader_cents = []
    for username in usernames:
        sum_leader_cents.append(centrality_per_topic(G,username=username,mean=False))
        mean_leader_cents.append(centrality_per_topic(G,username=username,mean=True))
    plot_dual_centralities(sum_overall_topic_centralities,sum_leader_cents,expand=False,usernames=usernames,mean=False)
    plot_dual_centralities(sum_overall_topic_centralities,sum_leader_cents,expand=True,usernames=usernames,mean=False)
    plot_dual_centralities(mean_overall_topic_centralities,mean_leader_cents,expand=False,usernames=usernames,mean=True)
    plot_dual_centralities(mean_overall_topic_centralities,mean_leader_cents,expand=True,usernames=usernames,mean=True)
