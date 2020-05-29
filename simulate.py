import msprime
import numpy as np
pop_configs = [
    msprime.PopulationConfiguration(sample_size=100, growth_rate = 0),
    msprime.PopulationConfiguration(sample_size=100, growth_rate = 0)]
divergence_time = 200
divergence_event1 = msprime.MassMigration(time = divergence_time, source = 1, dest = 0, proportion = 1)
Ne = 400
length = 10
recomb = 0.5
mutation_rate = 0.05
pops = [range(0,100),range(100,200),range(200,300)]
ts_ref = msprime.simulate(population_configurations = pop_configs,length = length,Ne = Ne,mutation_rate=mutation_rate,
                      demographic_events = [divergence_event1],
                      random_seed = 2, recombination_rate = recomb)
def simulate(data,mutations,divergence_time,length,pop_configs,pops,Ne,recomb,divergence2,prior_parameters=(0.1,10),iters=100):
    ref_stat = (data.f2([pops[0],pops[1]]),data.Fst([pops[0],pops[1]]),data.diversity(),data.Tajimas_D(),data.segregating_sites())
    p = len(ref_stat)
    samples = np.zeros((iters,2))
    discrepancies = np.zeros((iters,p))
    for i in range(iters):
        if mutations == "exponential":
            mutation_rate_sim = np.random.exponential(prior_parameters[0])
        elif mutations == "uniform":
            mutation_rate_sim = np.random.uniform(prior_parameters[0][0],prior_parameters[0][1])
        else:
            mutation_rate_sim = mutations
        if divergence_time == "exponential":
            divergence_time_sim = min(np.random.exponential(prior_parameters[1]),1000)
        elif divergence_time == "uniform":
            divergence_time_sim = min(np.random.uniform(prior_parameters[1][0],prior_parameters[1][1]),1000)
        else:
            divergence_time_sim = divergence_time
        divergence_event1 = msprime.MassMigration(time = divergence_time_sim, source = 1, dest = 0, proportion = 1)
        ts_sim = msprime.simulate(population_configurations = pop_configs,Ne=Ne,length=length, mutation_rate=mutation_rate_sim,
                 demographic_events = [divergence_event1],recombination_rate = recomb)
        sim_stat = (ts_sim.f2([pops[0],pops[1]]),ts_sim.Fst([pops[0],pops[1]]),ts_sim.diversity(),ts_sim.Tajimas_D(),ts_sim.segregating_sites())
        samples[i] = ((mutation_rate_sim,divergence_time_sim))
        d = np.subtract(sim_stat,ref_stat)
        discrepancies[i] = d
    return(samples,discrepancies)
attempt1 = simulate(ts_ref,0.05,"uniform",length,pop_configs,pops,Ne,recomb,divergence_event2,prior_parameters=(0.2,(100,300)),iters=5000)
np.save('simulations',a)
