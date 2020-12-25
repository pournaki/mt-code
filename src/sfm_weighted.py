import networkx as nx
import numpy as np
from tqdm import tqdm 

def social_feedback_simu(G,
                         Q,
                         alpha=0.01,
                         beta_opinion=3.0,
                         beta_neighborselection=3.0,
                         max_steps=10000,  
                         progressbar=True,
                         ):
    
    """
    Simulate the social feedback model [Banisch and Olbrich, 2017] on a graph and 
    return the time it takes the system to reach consensus. If the consensus is not
    reached in the max_step time, returns 0.

    Parameters
    ----------
    G : a graph as adjacency matrix
    Q : array
        Initial conditions (N x i) matrix, where N is the number of nodes and i
        the number of allowed opinions    
    alpha : float
        Learning rate (default: 0.01)
    beta_opinion : float
        Boltzmann action parameter for opinion selection (default: 3.0)
    beta_neighborselection : float
        Boltzmann action parameter for neighbor selection (default: 3.0)
    max_steps : int
        Number of steps before stopping the simulation (default: 10000)
    progressbar : boolean
        Show a tqdm progress bar

    Returns
    ----------
    t_c : int
        Time it took the system to reach consensus or 0 if it didn't reach it.
    """

    graph_adj = np.array(G)
    N = len(graph_adj)

    alpha = alpha  # learning rate
    steps = max_steps

    Q = np.copy(Q)

    n_coms = Q.shape[1]

    # get all node's neighbors and the probability that they are chosen
    neighbordict = {}
    for node in range(N):
        neighbors = list(graph_adj[node].nonzero()[0])
        weights = list(graph_adj[node][neighbors])
        beta = beta_neighborselection
        probs = np.exp(beta * np.array(weights)) / np.sum(np.exp(beta * np.array(weights)))
        neighbordict[node] = {}
        neighbordict[node]["neighbors"] = neighbors
        neighbordict[node]["weights"] = weights
        neighbordict[node]["probs"] = probs

    # assert the initial opinions vector
    opinionList = []
    for i in range (N):
        opinionList.append(np.argmax(Q[i,:]))

    # get the initial belongingness vector
    belongingnessList = []
    for n in range(N):
        com           = opinionList[n]
        friends       = np.where(np.array(opinionList) == com)[0]
        friends_edges = np.sum(graph_adj[n][friends])
        deg           = np.count_nonzero(graph_adj[n])
        belongingness      = friends_edges / deg
        belongingnessList.append(belongingness)

    # and the minimum belongingness
    min_bel = min(belongingnessList)

    opinionarrayList = []
    #minbelList = []

    opinionarrayList.append(opinionList)
    #minbelList.append(min_bel)

    # we want to know the times of nash equilibria
    t_c = 0
    t_ns = []

    if progressbar == False:
        dis = True
    else:
        dis = False


    def part_to_mag(part):
        N = len(part)
        if set(part) == {'1'}:
            return 1
        elif set(part) == {'0'}:
            return -1
        else:
            part = np.array(part)
            part[part==0]=-1
            return float(np.sum(part))/float(N)
    
    magnetizations = []

    magnetizations.append(part_to_mag(opinionList))
    
    
    if beta_opinion > 0:
        # evaluate the SFM
        for step in tqdm(range(max_steps), disable=dis):

            # pick one agent at random
            a1 = np.random.choice(range(N))

            # boltzmann action
            Qexp = np.exp(beta_opinion * Q[a1,:]) / np.sum(np.exp(beta_opinion * Q[a1,:]))
            expression = np.random.choice(np.arange(n_coms), replace=True, p=Qexp)

            # --- reaction part differs depending on setting ---
            a1sneighbors = neighbordict[a1]["neighbors"]
            probs = neighbordict[a1]["probs"]
            a2 = np.random.choice(a1sneighbors, p=probs)

            Qexp2 = np.exp(beta_opinion * Q[a2,:]) / np.sum(np.exp(beta_opinion * Q[a2,:]))
            reaction = np.random.choice(np.arange(n_coms), replace=True, p=Qexp2)

            reward = int(expression==reaction)
            reward = (reward * 2) - 1
            Q[a1,expression] = (1-alpha) * Q[a1,expression] + alpha*reward

            # now find out the new opinions
            opinionList = []
            for i in range (N):
                opinionList.append(np.argmax(Q[i,:]))            
            opinionarrayList.append(opinionList)

            magnetizations.append(part_to_mag(opinionList))
            
            # are we at consensus? if yes, then break the loop!
            # if len(set(np.argmax(Q, axis=1))) == 1:
            #     t_c += step
            #     break    

    # this will correspond to emperature 0
    elif beta_opinion == 0:
        # evaluate the SFM
        for step in tqdm(range(max_steps), disable=dis):

            # pick one agent at random
            a1 = np.random.choice(range(N))

            # best response
            expression = np.random.choice(np.flatnonzero(Q[a1,:] == Q[a1,:].max()))

            # --- reaction part differs depending on setting ---
            a1sneighbors = neighbordict[a1]["neighbors"]
            # probs = neighbordict[a1]["probs"]
            # a2 = np.random.choice(a1sneighbors, p=probs)
            a2 = np.random.choice(a1sneighbors)

            reaction = np.random.choice(np.flatnonzero(Q[a2,:] == Q[a2,:].max()))

            reward = int(expression==reaction)
            reward = (reward * 2) - 1
            Q[a1,expression] = (1-alpha) * Q[a1,expression] + alpha*reward

            # now find out the new opinions
            opinionList = []
            for i in range (N):
                opinionList.append(np.argmax(Q[i,:]))            
            opinionarrayList.append(opinionList)

            # are we at consensus? if yes, then break the loop!
            # if len(set(np.argmax(Q, axis=1))) == 1:
            #     t_c += step
            #     break    

            magnetizations.append(part_to_mag(opinionList))
            
    return opinionarrayList #,magnetizations