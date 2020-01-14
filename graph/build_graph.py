import sys
import operator
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

class Graph(object):
    '''
    Initiates the networkx graph, houses visualization, as well as some quick/dirty analysis.
    
    :param usernames: A list of strings, corresponding to the twitter usernames stored in `/data`
    '''
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

    def draw_graph(self,save=False):
        G = self.G
        print("--- Adding colours and labels ---")
        colors = []
        legend = set()
        labels = {}
        retweet_labels = {}
        for node in G.nodes():
            attributes = G.nodes[node]
            if 'type' in attributes:
                if attributes['type'] == 'retweet':
                    retweet_labels[node] = node
                    colors.append('#79BFD3')
                elif attributes['type'] == 'tweet':
                    cluster = self.__return_colour(attributes["lda_cluster"])
                    legend.add(cluster)
                    colors.append(cluster[0])
                elif attributes['type'] == 'user':
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
                alpha=0.75,
                node_size=5,
                width=0.3,
                arrows=False
        )
        nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='w')
        plt.legend(handles=self.__return_legends(legend),loc="best")
        if save: plt.savefig("../visualizations/graph_{}_tweets_{}_retweeters_{}_retweets_{}_topics.pdf".format(self.num_tweets,self.num_retweeters,self.num_retweets,len(legend)-2))
        plt.show()

    def __return_legends(self,legend):
        legends = [Line2D([0], [0], marker='o', color='w', label='Party Leader',markerfacecolor='r', markersize=10),Line2D([0], [0], marker='o', color='w', label='Retweet',markerfacecolor='#79BFD3', markersize=10)]
        legend = sorted(legend, key=lambda tup: tup[1])
        for color,cluster_num in legend:
            legends.append(Line2D([0], [0], marker='o', color='w', label='Topic {}'.format(cluster_num), markerfacecolor=color, markersize=10))
        return legends
        

    def build_graph(self,username):
        twitter_df = pd.read_csv("../data/{}_data.csv".format(username))
        twitter_df = twitter_df.sample(n=min(400,len(twitter_df)),random_state=4)
        retweet_df = pd.read_csv("../data/{}_retweets.csv".format(username))
        retweet_df = retweet_df[retweet_df['original_tweet_id'].isin(twitter_df['id'])] # if we're only taking 20 tweets find all the retweets for those 20
        G = nx.MultiDiGraph()
        G.add_node(username,type='user')
        # Instantiate a new MultiDiGraph (graph is directional + there could potentially be multip edges between a pair of nodes)
        # add tweet nodes
        nodes = twitter_df.set_index('id').to_dict('index').items()
        G.add_nodes_from(nodes)
        for _,row in twitter_df.iterrows():
            G.add_edge(username,row['id'],weight=row['favorite_count'])
        # add retweet user nodes (those who retweeted the original tweets) multipl
        user_nodes = retweet_df.drop_duplicates(subset ="original_author") 
        user_nodes = user_nodes.set_index('original_author').to_dict('index').items()
        G.add_nodes_from(user_nodes)
        for _,row in retweet_df.iterrows():
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


    def __return_colour(self,aNum):
        colours = [ "#006816",
                    "#8d34e4",
                    "#c9a738",
                    "#0163d0",
                    "#ee5700",
                    "#00937e",
                    "#ff4284",
                    "#4b5400",
                    "#ea80ff",
                    "#9f0040"]
        assert aNum < len(colours) 
        return colours[aNum], aNum+1
        
    def max_degree_tweet(self):
        #TODO
        G = self.G
        tweets = (node for node in G if G.node[node]['type']=='tweet')
        degree_list = list(G.degree(tweets))
        return degree_list

if __name__ == '__main__' :
    # Read in CSV file for that twitter user (these are the original tweets)
    G = Graph(sys.argv[1:])
    G.draw_graph(save=True)