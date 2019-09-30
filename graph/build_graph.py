#!/usr/bin/python
import sys
import operator
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd

class Graph(object):
    def __init__(self,usernames):
        self.num_retweeters = 0
        self.num_tweets = 0
        self.num_retweets = 0
        self.usernames = usernames
        self.G = nx.MultiDiGraph()
        self.title = ""
        for username in usernames:
            self.title += "{}_".format(username)
            user_graph = self.build_graph(username)
            self.G = nx.compose(self.G,user_graph)

    def draw_graph(self):
        G = self.G
        title = "../visualizations/{}graph.png".format(self.title)
        print("--- Adding colours and labels ---")
        colors = []
        labels = {}
        retweet_labels = {}
        for node in G.nodes():
            attributes = G.node[node]
            if 'type' in attributes:
                if G.node[node]['type'] == 'retweet':
                    retweet_labels[node] = node
                    colors.append('#79BFD3')
                elif G.node[node]['type'] == 'tweet':
                    colors.append('#FFDE03')
                elif G.node[node]['type'] == 'user':
                    labels[node] = node
                    colors.append('red')
        
        print("--- Laying out {} nodes and {} edges ---".format(len(G.nodes()),G.number_of_edges()))
        print("--- {} tweets, {} retweeters, {} retweets ---".format(self.num_tweets,self.num_retweeters,self.num_retweets))

        plt.figure(figsize=(30, 30))
        # use graphviz to find radial layout
        pos = graphviz_layout(G, prog="sfdp")
        # pos = nx.spring_layout(G, k=0.15, iterations=20)
        # draw nodes, coloring by rtt ping time
        nx.draw(G, pos,
                node_color=colors,
                with_labels=False,
                alpha=0.5,
                node_size=80,
                arrowsize=10,
                arrowstyle='fancy')
        nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='b')
        # nx.draw_networkx_labels(G,pos,retweet_labels,font_size=8,font_color='g')
        plt.savefig(title, bbox_inches="tight")
        # plt.show()

    def build_graph(self,username):
        twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
        retweet_df = pd.read_csv("../data/{}_retweets.csv".format(username))
        retweet_df = retweet_df[retweet_df['original_tweet_id'].isin(twitter_df['id'])] # if we're only taking 20 tweets find all the retweets for those 20
        G = nx.MultiDiGraph()
        G.add_node(username,type='user')
        # Instantiate a new MultiDiGraph (graph is directional + there could potentially be multip edges between a pair of nodes)
        # add tweet nodes
        nodes = twitter_df.set_index('id').to_dict('index').items()
        G.add_nodes_from(nodes)
        for index,row in twitter_df.iterrows():
            G.add_edge(username,row['id'])
        # add retweet user nodes (those who retweeted the original tweets) multipl
        user_nodes = retweet_df.drop_duplicates(subset ="original_author") 
        user_nodes = user_nodes.set_index('original_author').to_dict('index').items()
        G.add_nodes_from(user_nodes)
        for index,row in retweet_df.iterrows():
            G.add_edge(row['original_tweet_id'],row['original_author'])
        self.num_retweeters += len(user_nodes)
        self.num_tweets += len(twitter_df)
        self.num_retweets += len(retweet_df)
        return G

    def to_adjecency_matrix(self):
        G = self.G
        title = self.title
        matrix = nx.to_numpy_matrix(G)
        pd.DataFrame(matrix).to_csv(title+"adj_matrix.csv")

    def get_density(self):
        density = nx.density(self.G)
        print("The percentage of edges/possible edges is {0:.4f}%: ".format(density*100))
        return density
        
    def max_degree_tweet(self):
        #TODO
        G = self.G
        tweets = (node for node in G if G.node[node]['type']=='tweet')
        degree_list = list(G.degree(tweets))
        return degree_list

if __name__ == '__main__' :
    # Read in CSV file for that twitter user (these are the original tweets)
    G = Graph(sys.argv[1:])
    G.draw_graph()