import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def read_graph(input, weighted=False, directed=False, delim=None):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph(), delimiter=delim)
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph(), delimiter=delim)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not directed:
            G = G.to_undirected()
    return G


def visualize_graph(G):
    '''
    Graphically show the network
    :param G:
    :return:
    '''
    nx.draw_spring(G, with_labels=True)
    plt.show()


def random_barabasi_albert_probabilistic_graph(n=500, m=5, probrange=(0, 1)):
    G = nx.barabasi_albert_graph(n, m)
    for n1, n2 in G.edges():
        G[n1][n2]['prob'] = np.random.uniform(probrange[0], probrange[1])
    return G


def random_erdos_renyi_probabilistic_graph(n=500, p=0.1, probrange=(0, 1)):
    G = nx.erdos_renyi_graph(n, p)
    for n1, n2 in G.edges():
        G[n1][n2]['prob'] = np.random.uniform(probrange[0], probrange[1])
    return G


def save_probabilistic_graph(G, output):
    outputf = open(output, 'w')
    for n1, n2, data in G.edges(data=True):
        outputf.write(str(n1) + ' ' + str(n2) + ' ' + str(data['prob'])+'\n')
    outputf.close()


def read_probabilistic_graph(input, weighted=False, directed=False):
    '''
        Reads the input network in networkx.
        Sample:
        1 2 0.8
        2 3 0.9
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('prob', float),('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, data=(('prob', float),), create_using=nx.DiGraph())
    if not directed:
        G = G.to_undirected()
    return G


def synthesize_probabilistic_graph(graphs, alpha=1.0):
    # type: (list) -> nx.Graph
    """
    Synthesize discrete graphs into a probabilistic graph. The probability decays exponentially over the time.
    :param graphs: Discrete graphs
    :param alpha: The decay ratio
    :return: A probabilistic graph, with edge has a attr "prob".
    """
    tau = len(graphs)
    probG = nx.Graph()
    Edges = {}
    for t in range(tau):
        graph = graphs[t]
        for edge in graph.edges():
            if tau > 1:
                Edges[edge] = Edges.get(edge, 1.0)*(1.0 - np.exp(-alpha*(tau-1-t)/(tau-1)))
            else:
                Edges[edge] = Edges.get(edge, 1.0)
    for item in Edges.keys():
        probG.add_edge(item[0], item[1], prob=1.0-Edges.get(item))
    return probG


def graph_trim(G, deg=1):
    # type: (nx.Graph) -> nx.Graph
    """
    Recursively remove the nodes whose degree is 1
    :param G: the graph
    :return: a new graph
    """
    newG = G.copy()
    stub_nodes = [node for node in newG.nodes() if newG.degree(node) <= deg]
    while len(stub_nodes)>1:
        newG.remove_nodes_from(stub_nodes)
        stub_nodes = [node for node in newG.nodes() if newG.degree(node) <= deg]
    return newG


def main():
    ERG = random_erdos_renyi_probabilistic_graph()
    save_probabilistic_graph(ERG, './data1/test_ERGraph.txt')


if __name__ == '__main__':
    main()
