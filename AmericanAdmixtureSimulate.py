import numpy as np
import sys
from numpy import savetxt
seed = int(sys.argv[1])

mu = 1.25e-8
rho = 1e-8
nbp = 1e8
N0 = 7310
Thum = 5920
Naf = 14474
Tooa = 2040
Nb = 1861
mafb = 1.5e-4
Teu = 920
Neu = 1031
Nas = 554
mafeu = 2.5e-5
mafas = 7.8e-6
meuas = 3.11e-5
reu = 0.0038
ras = 0.0048
Tadmix = 12
Nadmix = 30000
radmix = 0.05

refsamplesize=200
admsamplesize=200
pop_config = [msprime.PopulationConfiguration(sample_size=refsamplesize,initial_size=Naf,growth_rate=0.0),
              msprime.PopulationConfiguration(sample_size=refsamplesize,initial_size=Neu*exp(reu*Teu),growth_rate=reu),
              msprime.PopulationConfiguration(sample_size=refsamplesize,initial_size=Nas*exp(ras*Teu),growth_rate=ras),
              msprime.PopulationConfiguration(sample_size=admsamplesize,initial_size=Nadmix*exp(radmix*Tadmix),growth_rate=radmix)]

mig_mat = [[0,mafeu,mafas,0],[mafeu,0,meuas,0],[mafas,meuas,0,0],[0,0,0,0]]

admixture_event = [msprime.MassMigration(time=Tadmix,source=3,destination=0,proportion=1.0/6.0),
                   msprime.MassMigration(time=Tadmix+0.0001,source=3,destination=1,proportion=2.0/5.0),
                   msprime.MassMigration(time=Tadmix+0.0002,source=3,destination=2,proportion=1.0)]

census_event = [msprime.CensusEvent(time=Tadmix+0.0003)]

eu_event = [msprime.MigrationRateChange(time=Teu,rate=0.0),
            msprime.MassMigration(time=Teu+0.0001,source=2,destination=1,proportion=1.0),
            msprime.PopulationParametersChange(time=Teu+0.0002,initial_size=Nb,growth_rate=0.0,population_id=1),
            msprime.MigrationRateChange(time=Teu+0.0003,rate=mafb,matrix_index=(0,1)),
            msprime.MigrationRateChange(time=Teu+0.0003,rate=mafb,matrix_index=(1,0))]

ooa_event = [msprime.MigrationRateChange(time=Tooa,rate=0.0),
             msprime.MassMigration(time=Tooa+0.0001,source=1,destination=0,proportion=1.0)]

init_event = [msprime.PopulationParametersChange(time=Thum,initial_size=N0,population_id=0)]

events = admixture_event + census_event + eu_event + ooa_event + init_event

def generate_ancestry_table(ts,samples):
    census_ancestors = list(compress(list(range(ts.num_nodes)), ts.tables.nodes.flags==1048576))
    ancestry_table = ts.tables.link_ancestors(samples=samples,ancestors=census_ancestors)
    return ancestry_table
def generate_tracts(samples,ancestry_table,pop_id):
    ##Outputs a list of list tracts and a list of list of tract ids. Each list corresponds to ancestry of a sample
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
def sort_tracts(ts,sample_pops,anc_pops,plot=False):
    #Calls previous 2 functions and sorts tracts to make a list of lists of tract lengths,
    #sorted by ancestral population
    samples = []
    len_anc_pops = len(anc_pops)
    for population in sample_pops:
        samples += list(ts.samples(population=population))
    ancestry_table = generate_ancestry_table(ts,samples)
    pop_id = ts.tables.nodes.population
    anc_ratio = []
    L_split = [[] for i in anc_pops]
    lowerpop = min(anc_pops)
    tracts,tract_ids = generate_tracts(samples,ancestry_table,pop_id)
    for j in range(len(tracts)):
        for i in range(len_anc_pops):
            l = list(compress(tracts[j],[not bool(i) for i in np.array(tract_ids[j])-anc_pops[i]]))
            L_split[i].extend(l)
    return(L_split)
def calculate_stats(tracts,total_length,inc_ratio=True):
    #Calculates statistics on the output of sort_tracts
    if tracts:
        stats = [len(tracts),np.mean(tracts),np.var(tracts)]
    else:
        stats = [0,0,0]
    if inc_ratio:
        stats.append(sum(tracts)/total_length)
    return(stats)
def local_ancestry_stats(ts,sample_pops,anc_pops,plot=False):
    #calls all functions
    tracts_list = sort_tracts(ts,sample_pops,anc_pops,plot=plot)
    stats = []
    L = [tract for tracts in tracts_list for tract in tracts]
    total_length = sum(L)
    stats.extend(calculate_stats(L,total_length,inc_ratio=False))
    for tracts in tracts_list:
        stats.extend(calculate_stats(tracts,total_length,inc_ratio=True))
    return(stats)
def standard_stats(ts):
    samples = ts.samples()
    pops = [ts.samples(population=0),ts.samples(population=1),ts.samples(population=2),ts.samples(population=3)]
    stats = [ts.f2([pops[3],pops[0]]),ts.f2([pops[3],pops[1]]),ts.f2([pops[3],pops[2]]),
             ts.Fst([pops[3],pops[0]]),ts.Fst([pops[3],pops[1]]),ts.Fst([pops[3],pops[2]]),
             ts.f3([pops[3],pops[0],pops[1]]),ts.f3([pops[3],pops[0],pops[2]]),ts.f3([pops[3],pops[1],pops[2]]),
             ts.diversity(),ts.diversity(pops[3]),ts.Tajimas_D(),ts.segregating_sites()]
    return stats



def abc_sim(iterations):
    stats=np.zeros((iterations,28))
    samples=np.zeros((iterations,4))
    for i in range(iterations):
        af_alpha,eu_alpha,as_alpha = np.random.dirichlet((1, 1, 1), 1)[0]
        Tadmix = np.random.uniform(0,50)
        events[0] = msprime.MassMigration(time=Tadmix,source=3,destination=0,proportion=af_alpha)
        events[1] = msprime.MassMigration(time=Tadmix+0.0001,source=3,destination=1,proportion=eu_alpha)
        events[2] = msprime.MassMigration(time=Tadmix+0.0002,source=3,destination=2,proportion=1.0)
        events[3] = msprime.CensusEvent(time=Tadmix+0.0003)
        treeseq = msprime.simulate(population_configurations=pop_config,migration_matrix=mig_mat,
                           demographic_events=events,length=nbp,recombination_rate=rho,mutation_rate=mu,random_seed=seed)
        stats[i] = list(local_ancestry_stats(treeseq,[3],[0,1,2],plot=False))+standard_stats(treeseq)
        samples[i] = [af_alpha,eu_alpha,as_alpha,Tadmix]
    return(stats,samples)
    
    

dest = './AmericanAdmix_stats'+str(seed)+'.csv'
savetxt(dest, a, delimiter=',')
