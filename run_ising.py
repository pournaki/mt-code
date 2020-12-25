import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('T', type=float)
args = ap.parse_args()

def get_probability(energy1, energy2, temperature):
    return np.exp((energy1 - energy2) / temperature)

def get_energy(spins):
    return -np.sum(
        interaction * spins * np.roll(spins, 1, axis=0) +
        interaction * spins * np.roll(spins, -1, axis=0) +
        interaction * spins * np.roll(spins, 1, axis=1) +
        interaction * spins * np.roll(spins, -1, axis=1)
    )/2


def update(spins, temperature):
    spins_new = np.copy(spins)
    i = np.random.randint(spins.shape[0])
    j = np.random.randint(spins.shape[1])
    spins_new[i, j] *= -1

    current_energy = get_energy(spins)
    new_energy = get_energy(spins_new)
    
    if temperature > 0:
        if get_probability(current_energy, new_energy, temperature) > np.random.random():
            return spins_new
        else:
            return spins
    else:
        if new_energy < current_energy:
            return spins_new
        elif new_energy == current_energy:
            choosefrom = [spins,spins_new]
            choice = np.random.choice(2)
            return choosefrom[choice]
        else:
            return spins
        
shape = (30, 30)

T = Ts[args.T]

meanmags = []
finalmags = []

# Interaction (ferromagnetic if positive, antiferromagnetic if negative)
interaction = 1

runs = 100
steps = 1000000
transient = 200000

for i in range (runs):
    # random incons
    spins = np.random.choice([-1, 1], size=shape)
    Ms = []
    for i in range (steps):
        spins = update(spins=spins, temperature=T)
        M = np.sum(spins)
        Ms.append(M)
        
    # transform to np array
    Ms = np.array(Ms)
    # cut off the transient
    Ms = Ms[transient:]
    # get the magnetization
    Ms = Ms / (30*30)
    finalmag = np.abs(Ms[-1])
    # take the absolute value
    Ms_abs = np.abs(Ms)
    # take the mean over time
    meanmag = np.mean(Ms_abs)
    # append
    meanmags.append(meanmag)
    finalmags.append(finalmag)
    
meanmags = np.array(meanmags)
finalmags = np.array(finalmags)

np.savetxt("./mm_ising_T{0:.2f}.txt".format(T), meanmags, fmt="%.10f")
np.savetxt("./fm_ising_T{0:.2f}.txt".format(T), finalmags, fmt="%.10f")
