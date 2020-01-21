import sys
from build_graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

NUM_TOPICS = 7
COLOURS = ["#006816", "#8d34e4", "#c9a738", "#0163d0", "#ee5700",
           "#00937e", "#ff4284", "#4b5400", "#ea80ff", "#9f0040"]


def centrality_per_topic(G, username=None, measure='mean'):
    """
        Calculates the eigenvector centrality for a network G, and then takes the sums/averages the centrality of tweets grouped by topic. 
        Parameters
        ----------
        :param G: The networkx Graph to calculate centrality from

        :param username: `optional` if present only uses that user's tweets when calculating the mean.

        :param measure: `optional` Options are ('mean','sum','zscore'). Choooses how to represent the aggregate zscores
    """
    assert (measure in ["mean","sum","zscore"]), "Unknown measure used, must be one of mean, sum, or zscore."
    netx_graph = G.G.copy()
    # centrality = nx.eigenvector_centrality_numpy(netx_graph)
    centrality = nx.eigenvector_centrality_numpy(netx_graph)
    topic_centrality = dict((topic, []) for topic in range(NUM_TOPICS))
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
    all_topic_centralities = np.concatenate(
        [centrality for _, centrality in topic_centrality.items()]).ravel()
    average_centrality = np.mean(all_topic_centralities)
    std_dev = np.std(all_topic_centralities)

    for topic, centrality_list in topic_centrality.items():
        if measure == 'zscore':
            # Calculate the z-score for each tweet of topic i relative to the average tweet centrality for the entire graph (or subgraph give).
            zscores = (np.array(centrality_list)-average_centrality)/std_dev
            # Calcualte the mean of the z-scores.
            topic_centrality[topic] = np.mean(zscores)
        elif measure == 'sum':
            topic_centrality[topic] = np.sum(centrality_list)
        else:
            topic_centrality[topic] = np.mean(centrality_list)
    return topic_centrality


def plot_dual_centralities(overall_net_cent, individaul_cents, expand=False, usernames=None, measure='mean'):
    """
        Plots the "overall network centrality" (the centralities of every tweet of a certain topic) for each topic over the 
        "individual network centrality" (the centralities of every tweet of a certain topic for a certain user)
        Parameters
        ----------
        :param overall_net_cent: A dictionary where each key is a topic, and the value is the sum/mean centrality for every tweet in the network of that topic. 

        :param individaul_cents: A list of dictionaries, where each element contains the centrality dictionary for each topic tweeted about by that user.

        :param expand: `optional` If true it plots all of the different users as with their individual centrality scores. Otherwise means all of them per topic.

        :param usernames: `optional` If expand is true this should correspond to the order of usernames in the individual_cents list. For plot legend purposes.

        :param measure: `optional` Options are ('mean','sum','zscore'). For title/filename purposes.
    """
    assert (measure in ["mean","sum","zscore"]), "Unknown measure used, must be one of mean, sum, or zscore."
    overall_net_cent = sorted(overall_net_cent.items(), key=lambda tup: tup[0])
    # get rid of topic -1 (empty tweets)
    overall_net_cent = overall_net_cent[1:] if overall_net_cent[0][0] == -1 else overall_net_cent
    leader_cents = []
    for leader_cent in individaul_cents:
        leader_cent = sorted(leader_cent.items(), key=lambda tup: tup[0])
        # get rid of topic -1 (empty tweet)
        leader_cent = leader_cent[1:] if leader_cent[0][0] == - \
            1 else leader_cent
        topics, centralities = map(list, zip(*leader_cent))
        centralities = np.nan_to_num(np.array(centralities))
        leader_cents.append(centralities)
    topics, average_overall_cents = map(list, zip(*overall_net_cent))
    if measure == "zscore":
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
        ax.set_xlabel('Topic Centrality for Individual\'s Network ({})'.format(measure), labelpad=125)
        ax.set_ylabel('Overall Network Topic Centrality ({})'.format(measure), labelpad=165)
    else:
        plt.xlabel('Topic Centrality for Individual\'s Network ({})'.format(measure))
        plt.ylabel('Overall Network Topic Centrality ({})'.format(measure))
    plt.title('Overall Importance of Topic/Importance of Topic to Individual ({})\n'.format(measure))
    plt.grid(True)
    if expand and usernames:
        assert len(usernames) == len(leader_cents)
        for i, (username, centralities) in enumerate(zip(usernames, leader_cents)):
            plt.scatter(centralities, average_overall_cents,
                        alpha=0.5, label=username, c=COLOURS[i])
            for j, txt in enumerate(topics):
                plt.annotate("{}".format(txt+1),
                             (centralities[j], average_overall_cents[j]))
            plt.legend()
        plt.savefig(
            "../visualizations/centrality_charts/{}_opposing_centrality_chart_(expanded).png".format(measure))
        plt.clf()
    else:
        average_individual_cents = np.mean(leader_cents, axis=0)
        plt.scatter(average_individual_cents, average_overall_cents, alpha=0.5)
        for j, txt in enumerate(topics):
            plt.annotate("Topic {}".format(
                txt+1), (average_individual_cents[j], average_overall_cents[j]))
        plt.savefig(
            "../visualizations/centrality_charts/{}_opposing_centrality_chart_(mean_of_leaders).png".format(measure))
        plt.clf()


if __name__ == "__main__":
    usernames = sys.argv[1:] if sys.argv[1:] else [
        "JustinTrudeau", "ElizabethMay", "theJagmeetSingh", "AndrewScheer", "MaximeBernier"]
    G = Graph(usernames, n=10)
    sum_overall_topic_centralities = centrality_per_topic(G, measure='sum')
    mean_overall_topic_centralities = centrality_per_topic(G, measure='mean')
    zscore_overall_topic_centralities = centrality_per_topic(
        G, measure='zscore')
    sum_leader_cents = []
    mean_leader_cents = []
    zscore_leader_cents = []
    for username in usernames:
        singl_leader_g = Graph([username])
        sum_leader_cents.append(centrality_per_topic(
            singl_leader_g, measure='sum'))
        mean_leader_cents.append(centrality_per_topic(
            singl_leader_g, measure='mean'))
        zscore_leader_cents.append(centrality_per_topic(
            singl_leader_g, measure='zscore'))
    plot_dual_centralities(sum_overall_topic_centralities,
                           sum_leader_cents, expand=False, usernames=usernames, measure='sum')
    plot_dual_centralities(sum_overall_topic_centralities,
                           sum_leader_cents, expand=True, usernames=usernames, measure='sum')
    plot_dual_centralities(mean_overall_topic_centralities,
                           mean_leader_cents, expand=False, usernames=usernames, measure='mean')
    plot_dual_centralities(mean_overall_topic_centralities,
                           mean_leader_cents, expand=True, usernames=usernames, measure='mean')
    plot_dual_centralities(zscore_overall_topic_centralities,
                           zscore_leader_cents, expand=False, usernames=usernames, measure='zscore')
    plot_dual_centralities(zscore_overall_topic_centralities,
                           zscore_leader_cents, expand=True, usernames=usernames, measure='zscore')
