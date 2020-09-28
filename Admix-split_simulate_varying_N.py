import msprime
import numpy as np
from itertools import chain, compress
from scipy.spatial import distance
from numpy import asarray
from numpy import savetxt

def ts_sim(t_a,t_as,alpha,N1,N2,N3):
    t_s = 100
    N1 = 200
    pop_configs = [
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N1),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N2),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N2),
        msprime.PopulationConfiguration(sample_size=50, growth_rate = 0,initial_size=N3)]

    demographic_events = [msprime.MassMigration(time = t_as, source = 2, dest = 1, proportion = 1),
                          msprime.MassMigration(time = t_a, source = 1, dest = 0, proportion = alpha),
                          msprime.MassMigration(time = t_a+0.001, source = 1, dest = 3, proportion = 1),
                          msprime.CensusEvent(time=t_a+0.002),
                          msprime.MassMigration(time = t_s, source = 3, dest = 0, proportion = 1)
                         ]       
    length = 248956422
    recomb = 1.14856e-08
    mutation_rate = 1.29e-08
    ts = msprime.simulate(population_configurations = pop_configs,length = length,mutation_rate=mutation_rate,
                      demographic_events = demographic_events, recombination_rate = recomb)
    return(ts)


def generate_ancestry_table(ts,samples):
    census_ancestors = list(compress(list(range(ts.num_nodes)), ts.tables.nodes.flags==1048576))
    ancestry_table = ts.tables.link_ancestors(samples=samples,ancestors=census_ancestors)
    return ancestry_table
def local_ancestry(samples,ancestry_table,pop_id):
    ancestry_table = np.array((ancestry_table.left,ancestry_table.right,ancestry_table.parent,ancestry_table.child)).T
    anc = ancestry_table[ancestry_table[:,0].argsort()]
    num_samples = max(samples) + 1
    diff_pop = False
    tract_pop = [0]*num_samples
    for j in range(len(samples)):
        sample = int(anc[j,3])
        tract_pop[sample] = pop_id[int(anc[j,2])]
    tracts = [[] for i in range(num_samples)]
    tract_ids = [[] for i in range(num_samples)]
    tract_start = [0 for i in range(num_samples)]
    sample_index = [0]
    tract_end = int(max(anc[:,1]))
    for k in range(np.shape(anc)[0]):
        sample = int(anc[k,3])
        current_pop = pop_id[int(anc[k,2])]
        diff_pop = current_pop - tract_pop[sample]
        if diff_pop:
            tracts[sample].append(anc[k,0]-tract_start[sample])
            tract_ids[sample].append(tract_pop[sample])
            tract_start[sample] = anc[k,0]
        if anc[k,1] == tract_end:
            tracts[sample].append(anc[k,1]-tract_start[sample])
            tract_ids[sample].append(tract_pop[sample])
        tract_pop[sample] = current_pop
    tracts = [i for i in tracts if i]
    tract_ids = [i for i in tract_ids if i]
    return(tracts,tract_ids)
#ts - tree sequence
#sample_pops - population id of samples for which local ancestry statistics are calculated
#anc_pops = ancestral population ids for which tract lengths are calculated (2 populations)
def local_ancestry_tracts(ts,sample_pops,anc_pops,plot=False):
    samples = []
    len_anc_pops = len(anc_pops)
    for population in sample_pops:
        samples += list(ts.samples(population=population))
    ancestry_table = generate_ancestry_table(ts,samples)
    pop_id = ts.tables.nodes.population
    anc_ratio = []
    L_split = [[] for i in anc_pops]
    lowerpop = min(anc_pops)
    tracts,tract_ids = local_ancestry(samples,ancestry_table,pop_id)
    for j in range(len(tracts)):
        for i in range(len_anc_pops):
            l = list(compress(tracts[j],[not bool(i) for i in np.array(tract_ids[j])-anc_pops[i]]))
            L_split[i].extend(l)
    return(L_split)
def calculated_stats(tracts,total_length,inc_ratio=True):
    stats = [len(tracts),np.mean(tracts),np.var(tracts)]
    if inc_ratio:
        stats.append(sum(tracts)/total_length)
    return(stats)
def local_ancestry_stats(ts,sample_pops,anc_pops,plot=False):
    tracts_list = local_ancestry_tracts(ts,sample_pops,anc_pops,plot=plot)
    stats = []
    L = [tract for tracts in tracts_list for tract in tracts]
    total_length = sum(L)
    stats.extend(calculated_stats(L,total_length,inc_ratio=False))
    for tracts in tracts_list:
        stats.extend(calculated_stats(tracts,total_length,inc_ratio=True))
    return(stats)
def ABCsimulate(iterations):
    stats=np.zeros((iterations,41))
    samples=np.zeros((iterations,6))
    for i in range(iterations):
        t = (np.random.uniform(1,95),np.random.uniform(1,95))
        t_a,t_as = (max(t),min(t))
        (N1,N2,N3) = (np.random.uniform(10,500),np.random.uniform(10,500),np.random.uniform(10,500))
        alpha = np.random.uniform(0,1)
        ts = ts_sim(t_a,t_as,alpha,N1,N2,N3)
        stats[i] = list(local_ancestry_stats(ts,[2],[0,3]))+list(local_ancestry_stats(ts,[1],[0,3]))+standard_stats(ts)
        samples[i] = [t_a,t_as,alpha,N1,N2,N3]
    return(stats,samples)
stats,samples = ABCsimulate(10000)
savetxt('./AdmixSplitStats.csv',stats,delimiter=',')
savetxt('./AdmixSplitSamples.csv',samples,delimiter=',')
