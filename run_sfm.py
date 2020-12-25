#!/usr/bin/python

import numpy as np
import networkx as nx
from sfm_weighted import social_feedback_simu

runs = 100

# load the adjacency matrix
G_mat = np.loadtxt("./data/kc_mat_weighted.txt", dtype=int)

# load initial conditions
# you can choose from the following:
# incon_literature.txt
# incon_onlyleaders.txt

incon = np.loadtxt("./data/incon_literature.txt", delimiter=',')

network = G_mat

N = len(network)

def compute_min_belongingness(graph_adj, partition):
    N = len(graph_adj)
    belongingnessList = []
    comsarray = partition
    for n in range(N):
        com           = comsarray[n]
        friends       = np.where(np.array(comsarray) == com)[0]
        friends_edges = np.sum(graph_adj[n][friends])
        neighbors     = np.nonzero(graph_adj[n])
        deg           = np.sum(graph_adj[n][neighbors])
        belongingness = float(friends_edges) / float(deg)
        belongingnessList.append(belongingness)
    return min(belongingnessList)

minbels = []
partitions = []

for i in range (runs):

    # uncomment this for random initial conditions
    #    incon = np.empty((34,2))
    #    for i in range (34):
    #        incon[i,0] = np.random.uniform(-0.1,0.1)
    #        incon[i,1] = np.random.uniform(-0.1,0.1)

    
    opinions = social_feedback_simu(G=network,
                                    Q=incon,
                                    alpha=0.1,
                                    beta_opinion=0,
                                    beta_neighborselection=0,
                                    max_steps=100000,
                                    update="normal",
                                    progressbar=False
                                         )

    minbel = compute_min_belongingness(G_mat,opinions[-1])
    minbels.append(minbel)

    # saves the final partition
    partstr = str(opinions[-1]).replace("[","").replace("]","").replace(",","").replace(" ","")
    partitions.append(partstr)

minbels = np.array(minbels)
partitions = np.array(partitions)

np.savetxt("./sfm_c.txt", minbels, fmt="%.2f")
np.savetxt("./sfm_p.txt", partitions, fmt="%s")