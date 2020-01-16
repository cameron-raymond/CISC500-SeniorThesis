import matplotlib.pyplot as plt
import networkx as nx

g = nx.Graph()
g.add_node( 'a' )
g.add_node( 'b' )
g.add_node( 'c' )
g.add_node( 'd' )
g.add_edge( 'a', 'b' )
g.add_edge( 'c', 'd' )

h = nx.Graph()
h.add_node( 'c' )
h.add_node( 'd' )

# Define the positions of a, b, c, d
positions = nx.spring_layout( g )


# Save the computed x and y dimensions for the entire drawing region of graph g
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# Produce image of graph g with a, b, c, d and some edges.
plt.savefig( "g2.png" )
#plt.show()

# Clear the figure.
plt.clf()

# Produce image of graph h with two nodes c and d which should be in
# the same positions of those of graph g's nodes c and d.
nx.draw( h, positions )

# Ensure the drawing area and proportions are the same as for graph g.
plt.axis( [ xlim[0], xlim[1], ylim[0], ylim[1] ] )

#plt.show()
plt.savefig( "h2.png" )