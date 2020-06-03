#! /usr/bin/python
# coding = utf-8

from network_util import read_probabilistic_graph
import networkx as nx

def probable_shortest_paths(probG, source, target, theta = 0.99):
    # type: (nx.Graph, int, int, float) -> list
    """
    @author Chenxu Wang

    There are maybe more than one paths in a probabilistic graph. What's more, the length of these paths may not
    equal, e.g., the lengths of some paths may be greater than others.
    This function enumerates all probable shortest paths between two vertexes, with a cover ratio is not less than a
    predefined threshold $\theta$.
    :param probG: A probabilistic graph, where each edge has an attribute 'prob' indicating the probability of edge
    existence
    :param source: the source vertex
    :param target: the destination vertex
    :return: a list of all probable shortest paths between source and target, and the cover ratio is greater than theta
    """
    all_remove_edges = []  # used to store all the removed edges in the algorithm
    probable_paths, minimum_prob_edges = shortest_paths_as_deterministic(probG, source, target)
    dependent_prob = 1.0
    temp = 1.0
    all_probable_paths = []
    for path, independent_prob in probable_paths:
        all_probable_paths.append((path, independent_prob, independent_prob*dependent_prob))
        temp *= 1 - independent_prob
    dependent_prob = temp
    for edge in minimum_prob_edges:
        all_remove_edges.append((edge[0], edge[1], {'prob':edge[2]}))
        probG.remove_edge(edge[0], edge[1])
    while len(probable_paths) > 0 and connectivity(all_probable_paths) < theta:
        # find the secondary shortest paths
        probable_paths, minimum_prob_edges = shortest_paths_as_deterministic(probG, source, target)
        for edge in minimum_prob_edges:
            all_remove_edges.append((edge[0], edge[1], {'prob':edge[2]}))
            probG.remove_edge(edge[0], edge[1])
        for path, independent_prob in probable_paths:
            all_probable_paths.append((path, independent_prob, independent_prob*dependent_prob))
            temp *= 1 - independent_prob
        dependent_prob = temp
    # restore the graph
    probG.add_edges_from(all_remove_edges)
    return all_probable_paths


def shortest_paths_as_deterministic(probG, source, target):
    # type: (nx.Graph, int, int) -> (list, set)
    """
    @author Chenxu Wang

    This function firstly calculates the shortest paths by treating a probabilistic graph as a deterministic one,
    and the calculates the probabilities of these paths by multiplying all the probabilities of the edges along a
    path. In addition, the function also returns the edges with minimum probabilities in each path, in order to
    remove these edges in next iterative.
    :param probG: The probabilistic graph
    :param source: the source vertex
    :param target: the destination vertex
    :return: a list of probable paths with each element containing the path and its independent probability,
    and a list of edges that have the minimum probabilities in the edge.
    """
    shortest_paths = nx.all_shortest_paths(probG, source, target)
    paths = []
    minimum_prob_edges = set()
    try:
        for path in shortest_paths:
            path_prob = 1.0
            minimum_prob = 1.0
            minimum_prob_edge = ()
            for i in range(len(path) - 1):
                edge_prob = probG.get_edge_data(path[i], path[i + 1])['prob']
                path_prob *= edge_prob
                if edge_prob < minimum_prob:  # obtain the edge on each path whose probability is minimum
                    minimum_prob = edge_prob
                    minimum_prob_edge = (path[i], path[i + 1], edge_prob)
            if len(minimum_prob_edge) > 0:
                minimum_prob_edges.add(minimum_prob_edge)
            paths.append((path, path_prob))
    except nx.NetworkXNoPath:
        paths = []
        minimum_prob_edges = set()
    return paths, minimum_prob_edges

def connectivity(probable_paths):
    # type: (list) -> float
    """
    @author Chenxu Wang

    This function calculates the cover ratio of the probable paths on the all probable shortest paths.
    Given a list of paths and their probability, the cover ratio is calculated as:
    $$ r = 1 - \prod_i 1 - p_i $$
    where p_i is the probability of the i-th path.
    :param probable_paths: a list of probable paths, each element is a tuple contains a probable path and its
    probability
    :return: the cover ratio
    """
    disconnect = 1.0
    for path in probable_paths:
        disconnect *= 1 - path[1]
    return 1.0 - disconnect


def probability_summary(probable_paths):
    """
    Calculate the sum of probabilities of all probable paths
    :param probable_paths:
    :return:
    """
    probsum = 0.0
    for path in probable_paths:
        probsum += path[2]
    return probsum


def main():
    probG = read_probabilistic_graph('./data/test.edgelist')
    # single_source_possible_shortest_path(probG, 4)
    # for edge in probG.edges(data=True):
    #     print edge
    print probG.order(), probG.size()
    paths = probable_shortest_paths(probG, 1, 4)
    for path in paths:
        print path
    print 'connectivity', connectivity(paths)


if __name__ == '__main__':
    main()
