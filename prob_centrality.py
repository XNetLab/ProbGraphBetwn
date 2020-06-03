#! /usr/bin/python
# coding = utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from network_util import *
from prob_shortest_paths import *
from util import fn_timer
import datetime


def probgraph_sample(probG):
    # type: (nx.Graph) -> nx.Graph
    '''
    Sample a probabilistic graph according to the edge probabilities.
    :param probG: a probabilistic graph
    :return: a deterministic graph as a possible world of probG
    '''
    rx = np.random.uniform(size=probG.size())
    detG = nx.Graph()
    # edgeprobs = nx.get_edge_attributes(probG,'prob')
    edgeprobs = list(probG.edges(data='prob'))
    detG.add_edges_from([(edgeprobs[i][0], edgeprobs[i][1]) for i in range(len(edgeprobs)) if edgeprobs[i][2]>rx[i]])
    return detG.to_undirected()


@fn_timer
def MC_betweenness(probG, eps=1e-2, delta=1e-3):
    # type: (nx.Graph, float, float) -> (dict, list)
    '''
    Calculate the expected betweenness centrality of nodes in a probabilistic graph by Monte Carlo methods.
    :param probG: a probabilistic graph
    :param eps: the accuracy of betweenness centrality
    :param delta: the confidence
    :return: btwn: the betweenness of each node calculated by Monte Carlo methods
            btwn_mean: the mean of the betweenness of all nodes at each step, this value is used to test whether it is
            convergent.
    '''
    r = int(2/(eps**2)*np.log(2/delta))
    print r
    btwn = {}
    btwn_mean = []
    for i in range(r):
        if i%1000 == 0:
            print i,
        sampled_G = probgraph_sample(probG)
        sample_betw = nx.betweenness_centrality(sampled_G, normalized=True)
        btwn = merge(btwn, sample_betw, merge_fn=lambda x,y:x+y)
        btwn_mean.append(np.sum(btwn.values())/probG.order()/(i+1))
    for key in btwn.keys():
        btwn[key] = btwn[key]/r
    return btwn, btwn_mean


@fn_timer
def PSP_betweenness(proG, logging=False):
    # type: (nx.proG) -> (dict, dict)
    '''
    Calculate the betweenness centrality using possible shortest paths.
    :param proG: the probabilistic graph
    :return: a tuple of dict, the betweenness of each node and the betweenness of each edge
    '''
    # node_pair_count = 0
    nodes = proG.nodes()
    N = len(nodes)
    node_btwn = {}
    edge_btwn = {}
    if logging:
        logfile = open('./data/pathlen.txt', 'w')
    for i in range(N):
        v = nodes[i]
        starttime = datetime.datetime.now()
        for j in range(i+1, N):
            # node_pair_count += 1.0
            u = nodes[j]
            shpaths = probable_shortest_paths(proG, v, u)
            conn = connectivity(shpaths)
            total_prob = probability_summary(shpaths)
            for path in shpaths:
                for node in path[0][1:-1]:
                    node_btwn[node] = node_btwn.get(node, 0.0) + path[2]*conn/total_prob
                for index in range(len(path[0])-1):
                    edge = (path[0][index], path[0][index+1])
                    edge_btwn[edge] = edge_btwn.get(edge, 0.0) + path[2] * conn / total_prob
            if logging:
                print >> logfile, i, j, v, u, len(shpaths)
        print i, (datetime.datetime.now()-starttime).seconds
    if logging:
        logfile.close()
    for key in node_btwn.keys():
        node_btwn[key] = node_btwn.get(key)*2.0/(N-1)/(N-2)
    for key in edge_btwn.keys():
        edge_btwn[key] = edge_btwn.get(key)*2.0/N/(N-1)
    return node_btwn, edge_btwn


@fn_timer
def VC_betweenness(probG, eps=1e-2, delta=1e-2):
    # type: (nx.Graph) -> dict
    '''
    Approximate the betweenness of nodes according to Vapnik-Chervonenkis dimension
    :param probG: the probabilistic graph
    :param eps: expected deviation of calculated results
    :param delta: confidence of calculated results
    :return: a dict, the betweenness of each node
    '''
    N = probG.order()
    VD = 100
    n = int(0.5/eps**2 * np.log2(VD-2) + np.log(1/delta))
    print n
    btwn = {}
    nodes = probG.nodes()
    for node in nodes:
        btwn[node] = 0.0
    i = 0
    for u, v in sample_node_pair(nodes, n):
        shpaths = probable_shortest_paths(probG, v, u)
        conn = connectivity(shpaths)
        total_prob = probability_summary(shpaths)
        for path in shpaths:
            for node in path[0][1:-1]:
                btwn[node] = btwn.get(node, 0.0) + path[2]*conn/total_prob
        if i%1000 == 0:
            print i, n, len(shpaths)
        i += 1
    for key in btwn.keys():
        btwn[key] = btwn.get(key)/n #*N/2.0/n/(N-2)
    return btwn


def sample_node_pair(nodes, N):
    # type: (list) -> tuple
    '''
    Randomly select N node pairs from the nodes list.
    :param nodes: a list of the nodes
    :return: a list of a tuple of two nodes
    '''
    maxindex = len(nodes)-1
    uindex = np.random.randint(low=0, high=maxindex, size=N)
    vindex = np.random.randint(low=0, high=maxindex, size=N)
    npairs = [(nodes[uindex[i]], nodes[vindex[i]]) for i in range(N) if uindex[i]!=vindex[i]]
    while len(npairs) < N:
        uindex = np.random.randint(low=0, high=maxindex)
        vindex = np.random.randint(low=0, high=maxindex)
        if uindex != vindex:
            npairs.append((nodes[uindex], nodes[vindex]))
    return npairs


def merge(d1, d2, merge_fn=lambda x,y:y):
    # type (dict, dict) -> dict
    """
    Merges two dictionaries, non-destructively, combining
    values on duplicate keys as defined by the optional merge
    function.  The default behavior replaces the values in d1
    with corresponding values in d2.  (There is no other generally
    applicable merge strategy, but often you'll have homogeneous
    types in your dicts, so specifying a merge technique can be
    valuable.)

    Examples:

    >>> d1
    {'a': 1, 'c': 3, 'b': 2}
    >>> merge(d1, d1)
    {'a': 1, 'c': 3, 'b': 2}
    >>> merge(d1, d1, lambda x,y: x+y)
    {'a': 2, 'c': 6, 'b': 4}

    """
    result = dict(d1)
    for k,v in d2.iteritems():
        if k in result:
            result[k] = merge_fn(result[k], v)
        else:
            result[k] = v
    return result


def local_psp_betweenness(probG, localnodes):
    # type: (nx.Graph, list) -> (dict, dict)
    """
    Calculate the betweenness centrality from the nodes in the localnodes to the others
    :param probG: the probabilistic graph
    :param localnodes: the list of local nodes
    :return: the local betweenness of nodes and edges
    """
    nodes = probG.nodes()
    lnodes = set(localnodes)
    node_btwn = {}
    edge_btwn = {}
    count = 0
    for v in localnodes:
        starttime = datetime.datetime.now()
        for u in nodes:
            if u in lnodes:
                continue
            count += 1.0
            shpaths = probable_shortest_paths(probG, v, u)
            conn = connectivity(shpaths)
            total_prob = probability_summary(shpaths)
            for path in shpaths:
                for node in path[0][1:-1]:
                    node_btwn[node] = node_btwn.get(node, 0.0) + path[2] * conn / total_prob
                for index in range(len(path[0]) - 1):
                    edge = (path[0][index], path[0][index + 1])
                    edge_btwn[edge] = edge_btwn.get(edge, 0.0) + path[2] * conn / total_prob
        print count, (datetime.datetime.now() - starttime).seconds
    for key in node_btwn.keys():
        node_btwn[key] = node_btwn.get(key) / count
    for key in edge_btwn.keys():
        edge_btwn[key] = edge_btwn.get(key) / count
    return node_btwn, edge_btwn


def local_psp_number(probG, localnodes):
    # type: (nx.Graph, list) -> (dict, dict)
    """
    Calculate the number of possible shortest paths from the nodes in the localnodes to the others
    :param probG: the probabilistic graph
    :param localnodes: the list of local nodes
    :return: a dict, with nodes as keys and the number of possilbe shortest paths as values
    """
    nodes = probG.nodes()
    lnodes = set(localnodes)
    node_nums = {}
    edge_nums = {}
    count = 0
    for v in localnodes:
        starttime = datetime.datetime.now()
        for u in nodes:
            count += 1.0
            if u in lnodes:
                continue
            shpaths = probable_shortest_paths(probG, v, u)
            conn = connectivity(shpaths)
            total_prob = probability_summary(shpaths)
            for path in shpaths:
                for node in path[0][1:-1]:
                    node_nums[node] = node_nums.get(node, 0.0) + 1.0
                for index in range(len(path[0]) - 1):
                    edge = (path[0][index], path[0][index + 1])
                    edge_nums[edge] = edge_nums.get(edge, 0.0) + 1.0
        print count, (datetime.datetime.now() - starttime).seconds
    node_total = sum(node_nums.values())
    for key in node_nums.keys():
        node_nums[key] = node_nums.get(key) / node_total
    edge_total = sum(edge_nums.values())
    for key in edge_nums.keys():
        edge_nums[key] = edge_nums.get(key) / edge_total
    return node_nums, edge_nums

