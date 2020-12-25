import numpy as np
import igraph as ig
import louvain
from tqdm import tqdm

meandegree1 = np.arange(0.1,2.05,0.05)
meandegree2 = np.arange(2.4,10.4,0.4)
meandegree3 = np.arange(11,506,6)
meandegreeI = np.append(meandegree1,meandegree2)
meandegree = np.append(meandegreeI,meandegree3)

Nlist = [100,200,300,400,500]

# fix N and use p
samples = 50

for N in (Nlist): 

    # initialize plotvals
    plotvals = []

    # initialize ps
    plist = meandegree / N    
    
    for p in tqdm(plist):

        vals = []
        
        if p <= 1:
        
            for i in range (samples):

                # draw a random graph with number of nodes N and probability p
                G = ig.Graph.Erdos_Renyi(n=N, p=p)

                # remove isolated nodes
                array = G.degree()
                array = np.array(array)
                todel = list(np.where(array==0)[0])
                G.delete_vertices(todel)    

                # compute louvain communities
                part = louvain.find_partition(G, louvain.ModularityVertexPartition)

                # get number of communities
                Nc = len(set(part.membership))

                # get modularity
                Q = part.modularity

                vals.append([N, Q, Nc])
        else:
            vals.append([N,0,0])
                
        vals = np.array(vals)
        Qmean = np.mean(vals[:,1])
        Qstdd = np.std(vals[:,1])
        Ncmean = np.mean(vals[:,2])
        Ncstdd = np.std(vals[:,2])
        
        plotvals.append([N, Qmean, Qstdd, Ncmean, Ncstdd])

    plotvals = np.array(plotvals)
    np.savetxt(f"./QRG_N{N}.txt",plotvals)