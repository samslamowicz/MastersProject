import msprime
import tskit
import sys
import numpy as np
from IPython.display import SVG
from itertools import chain, compress
from scipy.spatial import distance
import operator
from numpy import asarray
from numpy import savetxt
import math
import matplotlib.pyplot as plt

def ts_sim(t_a,t_s,alpha):
    t_as = 100
    N1 = 200
    pop_configs = [
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N1),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N1),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N1),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N1)]

    demographic_events = [msprime.MassMigration(time = t_s, source = 2, dest = 1, proportion = 1),
                          msprime.MassMigration(time = t_a, source = 1, dest = 0, proportion = alpha),
                          msprime.MassMigration(time = t_a+0.001, source = 1, dest = 3, proportion = 1),
                          msprime.CensusEvent(time=t_a+0.002),
                          msprime.MassMigration(time = t_as, source = 3, dest = 0, proportion = 1)
                         ]       
    length = 248956422
    recomb = 1.14856e-08
    mutation_rate = 1.29e-08
    ts = msprime.simulate(population_configurations = pop_configs,length = length,mutation_rate=mutation_rate,
                      demographic_events = demographic_events, recombination_rate = recomb)
    return(ts)
def local_ancestry(sample_id,ancestry_table,pop_id):
    sample_anc = np.array(list(compress(ancestry_table, list(ancestry_table[:,3]==sample_id))))
    sample_anc = sample_anc[sample_anc[:,0].argsort()]
    diff_pop = False
    chunk_pop = pop_id[int(sample_anc[0,2])]
    length_vec = []
    id_vec = []
    start = 0
    for i in range(np.shape(sample_anc)[0]):
        diff_pop = pop_id[int(sample_anc[i,2])] - chunk_pop
        if diff_pop:
            length_vec.append(sample_anc[i,0]-start)
            id_vec.append(chunk_pop)
            start = sample_anc[i,0]
        elif i == np.shape(sample_anc)[0]-1:
            length_vec.append(sample_anc[i,1]-start)
            id_vec.append(chunk_pop)
        chunk_pop = pop_id[int(sample_anc[i,2])]
    return(length_vec,id_vec)
def local_ancestry_stats(ts,pop,plot=False,return_l=False):
    samples = list(ts.samples(population=pop[0]))
    if len(pop) ==2:
        samples = list(ts.samples(population=pop[0])) + list(ts_ref.samples(population=pop[1]))
    #print(samples)
    census_ancestors = list(compress(list(range(ts.num_nodes)), ts.tables.nodes.flags==1048576))
    ancestry_table = np.array(ts.tables.link_ancestors(samples=samples,ancestors=census_ancestors))
    pop_id = ts.tables.nodes.population
    sample_stats = np.zeros((len(samples),8))
    l_t = []
    l_0_t = []
    l_1_t = []
    i = local_ancestry(samples[0],ancestry_table,pop_id)[1]
    lowerpop = min(i)
    for j in range(len(samples)):
        l,i = local_ancestry(samples[j],ancestry_table,pop_id)
        l_t.extend(l)
        if min(i)<lowerpop:
            lowerpop = min(i)
        #print(l)
        #print(samples[j])
        pop_0_l = list(compress(l,[bool(j) for j in np.array(i)-lowerpop]))
        pop_1_l = list(compress(l,[not bool(j) for j in np.array(i)-lowerpop]))
        l_0_t.extend(pop_0_l)
        l_1_t.extend(pop_1_l)
        #print(pop_0_l)
        #print(pop_1_l)
        if pop_0_l:
            mean_pop_0 = np.mean(pop_0_l)
            var_pop_0 = np.var(pop_0_l)
        else:
            mean_pop_0 = 0
            var_pop_0 = 0
        if pop_1_l:
            mean_pop_1 = np.mean(pop_1_l)
            var_pop_1 = np.var(pop_1_l)
        else:
            mean_pop_1 = 0
            var_pop_1 = 0
        mean_pop_1
        var_pop_1
        sample_stats[j] = [len(l),np.mean(l),mean_pop_0,mean_pop_1,np.var(l),var_pop_0,var_pop_1,sum(list(compress(l,[not bool(j) for j in i])))/sum(l)]
        #print(sample_stats[j])
    #print(sample_stats)
    if plot:
        plt.hist(l_t,bins=40,histtype = 'step')
        plt.hist(l_0_t,bins=40,histtype = 'step')
        plt.hist(l_1_t,bins=40,histtype = 'step')
        plt.show()
    stats = sample_stats.sum(axis=0)/len(samples)
    #print(stats)
    if return_l:
        return(stats,l_t,l_0_t,l_1_t)
    else:
        return(stats)
def standard_stats(ts):
    samples = ts.samples()
    pops = [ts.samples(population=0),ts.samples(population=1),ts.samples(population=2),ts.samples(population=3)]
    stats = [ts.f2([pops[0],pops[1]]),ts.f2([pops[0],pops[2]]),ts.f2([pops[0],pops[3]]),ts.f2([pops[1],pops[2]]),ts.f2([pops[1],pops[3]]),ts.f2([pops[2],pops[3]]),
             ts.Fst([pops[0],pops[1]]),ts.Fst([pops[0],pops[2]]),ts.Fst([pops[0],pops[3]]),ts.Fst([pops[1],pops[2]]),ts.Fst([pops[1],pops[3]]),ts.Fst([pops[2],pops[3]]),
             ts.f3([pops[2],pops[0],pops[3]]),ts.f3([pops[1],pops[0],pops[3]]),ts.f4([pops[0],pops[1],pops[2],pops[3]]),ts_ref.f4([pops[0],pops[2],pops[1],pops[3]]),
             ts.diversity(),ts.Tajimas_D(),ts.segregating_sites()]
    return stats
def ABCsimulate(iterations):
    stats=np.zeros((iterations,27))
    samples=np.zeros((iterations,2))
    for i in range(iterations):
        t = (np.random.uniform(1,95),np.random.uniform(1,95))
        t_a,t_s = (max(t),min(t))    
        ts = ts_sim(t_a,t_s,alpha)
        stats[i] = list(local_ancestry_stats(ts,[1,2]))+standard_stats(ts)
        samples[i] = [t_a,alpha]
    return(stats,samples)
stats,samples = ABCsimulate(50000)
savetxt('C:/Users/samue/OneDrive/Documents/Research/Code/Stat_comparison_test_Admix_split/stats1.csv', stats, delimiter=',')
savetxt('C:/Users/samue/OneDrive/Documents/Research/Code/Stat_comparison_test_Admix_split/samples1.csv', samples, delimiter=',')
