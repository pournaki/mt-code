# Finding meaningful communities in complex networks
This is the code repository for the main numerical results from my master's thesis on opinion-dynamics based approaches to community detection.

## Requirements
Make sure you have Python 3 installed on your system. Install the required libraries using: 

```
pip3 install requirements.txt
```

## Data

```
./data/finalsplit.txt --> "ground truth" partition from Zachary
./data/incon_literature.txt --> initial conditions according to initial leanings from Zachary's paper
./data/incon_onlyleaders.txt --> only the two leaders are initially opinionated
./data/kc_mat_unweighted.txt --> 78-edge unweighted Karate Club network 
./data/kc_mat_weighted.txt --> 78-edge weighted Karate Club, corresponds to symmetrized lower diagonal of Zachary's weighted adjacency matrix
```

## Modularity of random graphs
Compute the modularity of E-R random graphs for varying mean degree. The mean degree is varied between 0.1 and 500, the number of nodes between 100 and 500. For every graph couple of parameters, 50 samples are drawn. You can change these values according to your preference. The program saves a text file for every network size N, where every row corresponds to a mean degree and the columns correspond to N, <Q>, std(Q), <k>, std(k), where k is the number of communities.

Run it using 
```
python3 run_Q_randomgraphs.py
```

## Social feedback model
Simulate the social feedback model on the Karate Club network using different initial conditions. Per default, 100 runs are computed and the final partition for every run as well as the minimum cohesion is returned as text files. 

Run it using
```
python3 run_sfm.py
```

## Ising model on square lattice
Simulate the Metropolis-Hastings dynamics of the Ising model on the square lattice. Code adapted from https://github.com/carlosgmartin/ising. Returns the mean magnetization (after cutting off transient time) and final magnetization for different temperatures on $30 \times 30$ square lattice. 

Run it for different temperatures using for example
```
python3 run_ising.py --T=2.269
```
