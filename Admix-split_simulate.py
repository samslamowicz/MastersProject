import msprime
import numpy as np
from itertools import chain, compress
from scipy.spatial import distance
from numpy import asarray
from numpy import savetxt

print('test')
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


def generate_ancestry_table(ts,samples):
    census_ancestors = list(compress(list(range(ts.num_nodes)), ts.tables.nodes.flags==1048576))
    ancestry_table = np.array(ts.tables.link_ancestors(samples=samples,ancestors=census_ancestors))
    return ancestry_table
def local_ancestry(sample_id,ancestry_table,pop_id):
    sample_anc = np.array(list(compress(ancestry_table, list(ancestry_table[:,3]==sample_id))))
    sample_anc = sample_anc[sample_anc[:,0].argsort()]
    diff_pop = False
    tract_pop = pop_id[int(sample_anc[0,2])]
    tracts = []
    tract_ids = []
    tract_start = 0
    for i in range(np.shape(sample_anc)[0]):
        diff_pop = pop_id[int(sample_anc[i,2])] - tract_pop
        if diff_pop:
            tracts.append(sample_anc[i,0]-tract_start)
            tract_ids.append(tract_pop)
            tract_start = sample_anc[i,0]
        elif i == np.shape(sample_anc)[0]-1:
            tracts.append(sample_anc[i,1]-tract_start)
            tract_ids.append(tract_pop)
        tract_pop = pop_id[int(sample_anc[i,2])]
    return(tracts,tract_ids)
#ts - tree sequence
#sample_pops - population id of samples for which local ancestry statistics are calculated
#anc_pops = ancestral population ids for which tract lengths are calculated (2 populations)
def local_ancestry_stats(ts,sample_pops,anc_pops,plot=False,return_l=False):
    samples = []
    for population in sample_pops:
        samples += list(ts.samples(population=population))
    ancestry_table = generate_ancestry_table(ts,samples)
    pop_id = ts.tables.nodes.population
    anc_ratio = []
    L = [] #all tract lengths
    L_0 = []#all tract lengths with ancestry from ancestral population 0
    L_1 = []#all tract lengths with ancestry from ancestral population 1
    lowerpop = min(anc_pops)
    for j in range(len(samples)):
        l,i = local_ancestry(samples[j],ancestry_table,pop_id)
        L.extend(l)
        l_0 = list(compress(l,[bool(j) for j in np.array(i)-lowerpop]))
        l_1 = list(compress(l,[not bool(j) for j in np.array(i)-lowerpop]))
        L_0.extend(l_0)
        L_1.extend(l_1)
    if L_0:
        mean_L_0 = np.mean(L_0)
        var_L_0 = np.var(L_0)
    else:
        mean_L_0 = 0
        var_L_0 = 0
    if L_1:
        mean_L_1 = np.mean(L_1)
        var_L_1 = np.var(L_1)
    else:
        mean_L_1 = 0
        var_L_1 = 0
    stats = [len(L),len(L_0),len(L_1),np.mean(L),mean_L_0,mean_L_1,np.var(L),var_L_0,var_L_1,sum(L_0)/sum(L)]
    if plot:
        bin_list = list(range(0,int(1e8),int(5e5)))
        plt.hist(L,bins=bin_list,histtype = 'step')
        plt.hist(L_0,bins=bin_list,histtype = 'step')
        plt.hist(L_1,bins=bin_list,histtype = 'step')
        plt.xlim(xmin=0, xmax = 5e7)
        plt.show()
    if return_l:
        return(stats,L,L_0,L_1)
    else:
        return(stats)
def standard_stats(ts):
    samples = ts.samples()
    pops = [ts.samples(population=0),ts.samples(population=1),ts.samples(population=2),ts.samples(population=3)]
    stats = [ts.f2([pops[0],pops[1]]),ts.f2([pops[0],pops[2]]),ts.f2([pops[0],pops[3]]),ts.f2([pops[1],pops[2]]),ts.f2([pops[1],pops[3]]),ts.f2([pops[2],pops[3]]),
             ts.Fst([pops[0],pops[1]]),ts.Fst([pops[0],pops[2]]),ts.Fst([pops[0],pops[3]]),ts.Fst([pops[1],pops[2]]),ts.Fst([pops[1],pops[3]]),ts.Fst([pops[2],pops[3]]),
             ts.f3([pops[2],pops[0],pops[3]]),ts.f3([pops[1],pops[0],pops[3]]),ts.f4([pops[0],pops[1],pops[2],pops[3]]),ts.f4([pops[0],pops[2],pops[1],pops[3]]),
             ts.diversity(),ts.Tajimas_D(),ts.segregating_sites()]
    return stats
def ABCsimulate(iterations):
    stats=np.zeros((iterations,39))
    samples=np.zeros((iterations,2))
    for i in range(iterations):
        t = (np.random.uniform(1,95),np.random.uniform(1,95))
        t_a,t_s = (max(t),min(t))    
        alpha = np.random.uniform(0,1)
        ts = ts_sim(t_a,t_s,alpha)
        stats[i] = list(local_ancestry_stats(ts,[2],[0,3]))+list(local_ancestry_stats(ts,[1],[0,3]))+standard_stats(ts)
        samples[i] = [t_a,alpha]
    return(stats,samples)
ts = ts_sim(60,50,0.5)
samples = list(ts.samples(population=1))
t = generate_ancestry_table(ts,samples)
print(t)


stats,samples = ABCsimulate(1)
savetxt('./stats1.csv', stats, delimiter=',')
savetxt('./samples1.csv', samples, delimiter=',')
