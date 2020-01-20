import networkx as nx
from networkx.utils import py_random_state
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def draw_graph(G, save=False, file_type='png',use_pos=False):
        """
        Handles rendering and drawing the network.
        Parameters
        ----------
        :param G: a networkx graph. If present draws this graph instead of the one built in the constructor.

        :param save: `optional` A boolean. If true saves an image of the graph to `/visualizations` otherwise renders the graph.
        
        :param file_type: `optional` A string. If save flag is true it saves graph with this file extension.
        
        :param use_pos: `optional` A boolean. If true renders the graph using default positions of the entire graph. Otherwise calculates positions based on data used.
        """
        plt.figure(figsize=(30, 30))
        pos = graphviz_layout(G, prog="sfdp")

        # draw nodes, coloring by rtt ping time
        print("--- Drawing {} nodes and {} edges ---".format(len(G.nodes()), G.number_of_edges()))
        nx.draw(G, pos,
                with_labels=False,
                alpha=0.75,
                node_size=8,
                width=0.3,
                arrows=False
                )
        plt.show()

def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.
    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets

@py_random_state(3)
def powerlaw_cluster_graph(n, m, p, seed=None):
        """
        Adaption of the Holme and Kim algorithm for growing graphs with powerlaw
        degree distribution and approximate average clustering, as implemented by networkx
        Parameters
        ----------
        n : int
        the number of nodes
        m : int
        the number of random edges to add for each new node
        p : float,
        Probability of adding a triangle after adding a random edge
        seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        References
        ----------
        .. [1] P. Holme and B. J. Kim,
        "Growing scale-free networks with tunable clustering",
        Phys. Rev. E, 65, 026107, 2002.

        .. [2] https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py

        """

    G = nx.empty_graph(m) 
    repeated_nodes = list(G.nodes())  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = m               # next node is m
    while source < n:        # Now add the other n-1 nodes
        possible_targets = _random_subset(repeated_nodes, m, seed)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < m:  # add m-1 more new links
            if seed.random() < p:  # clustering step: add triangle
                neighborhood = [nbr for nbr in G.neighbors(target)
                                if not G.has_edge(source, nbr)
                                and not nbr == source]
                if neighborhood:  # if there is a neighbor without a link
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)  # add triangle
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue  # go to top of while loop
            # else do preferential attachment step if above fails
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1

        repeated_nodes.extend([source] * m)  # add source node to list m times
        source += 1
    return G

if __name__ == "__main__":
    new_g = powerlaw_cluster_graph(700,5,0)  
    draw_graph(new_g)