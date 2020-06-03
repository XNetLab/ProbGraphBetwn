import networkx as nx
import numpy as np
from network_util import *
from prob_shortest_paths import *
from prob_centrality import *
from util import fn_timer
import datetime


def test():
    ERGraph = read_probabilistic_graph('./data1/test_ERGraph.txt')
    MC_btwn, btwn_mean = MC_betweenness(ERGraph)
    for node, value in MC_btwn.items():
        print node, value
    node_btwn, edge_btwn = PSP_betweenness(ERGraph)
    for node, value in node_btwn:
        print node, value

if __name__ == '__main__':
    test()