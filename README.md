# MastersProject
import msprime
import sys
import numpy as np
import seaborn as sns
from IPython.display import SVG
import matplotlib.pyplot as plt

#Pairwise diversity
def treediversity(ts):
    stat = 0
    n = ts.num_samples
    for tree in ts.trees():
        f = 0
        for mutation in tree.mutations():
            x = tree.num_samples(mutation.node)
            f += x*(n-x)/(n*(n-1)/2)
        stat += f
    return(stat)

ts0 = msprime.simulate(5,mutation_rate = 0.1,recombination_rate=0,random_seed=5)
ts1 = msprime.simulate(5,mutation_rate = 0.1,recombination_rate=0.1,random_seed=6)
ts2 = msprime.simulate(10,mutation_rate = 0.1,recombination_rate=0.5,random_seed=7)
ts3 = msprime.simulate(105,mutation_rate = 2,recombination_rate=1,random_seed=8)
print(treediversity(ts0))
print(ts0.diversity())
print(treediversity(ts1))
print(ts1.diversity())
print(treediversity(ts2))
print(ts2.diversity())
print(treediversity(ts3))
print(ts3.diversity())

#Tajima's D
def TajimaD(ts):
    n = ts.num_samples
    a1 = sum([1/i for i in range(1,n)])
    b1 = (n+1)/(3*(n-1))
    c1 = b1 - 1/a1
    e1 = c1/a1
    a2 = sum([1/(i**2) for i in range(1,n)])
    b2 = 2*(n**2 + n + 3)/(9*n*(n-1))
    c2 = b2 - (n+2)/(a1*n) + a2/(a1**2)
    e2 = c2/(a1**2 + a2)
    S = ts.segregating_sites()
    theta_pi = treediversity(ts)
    theta_w = S/(a1)
    C = (e1*S + e2*S*(S-1))**(1/2)
    return((theta_pi-theta_w)/C)

pop_configs = [
    msprime.PopulationConfiguration(sample_size=5),
    msprime.PopulationConfiguration(sample_size=5)
]
M= np.array([
    [0,0.1],
    [0.2,0]
])
ts = msprime.simulate(population_configurations=pop_configs,migration_matrix=M,mutation_rate = 0.2,recombination_rate=0.5,random_seed=6)
print(ts.Tajimas_D())
print(TajimaD(ts))
ts1 = msprime.simulate(population_configurations=pop_configs,migration_matrix=M,mutation_rate = 0.4,recombination_rate=0.2,random_seed=6)
print(ts1.Tajimas_D())
print(TajimaD(ts1))
ts2 = msprime.simulate(population_configurations=pop_configs,migration_matrix=M,mutation_rate = 0.2,recombination_rate=0.5,random_seed=6)
print(ts2.Tajimas_D())
print(TajimaD(ts2))
pop_configs = [
    msprime.PopulationConfiguration(sample_size=5),
    msprime.PopulationConfiguration(sample_size=5),
    msprime.PopulationConfiguration(sample_size=3),
    msprime.PopulationConfiguration(sample_size=7)
]
M= np.array([
    [0,0.1,0.5,0.2],
    [0.2,0,0,0],
    [0.1,0,0,0.5],
    [0.1,0.1,0.1,0]
])
ts3 = msprime.simulate(population_configurations=pop_configs,migration_matrix=M,mutation_rate = 1,recombination_rate=0.3,random_seed=4)
print(ts3.Tajimas_D())
print(TajimaD(ts3))
ts4 = msprime.simulate(population_configurations=pop_configs,migration_matrix=M,mutation_rate = 0.1,recombination_rate=0.4,random_seed=6)
print(ts4.Tajimas_D())
print(TajimaD(ts4))
