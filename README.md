# contact-tracing-model

SEIR-T model

Dual and triad network topologies for modeling the effects of different "test and trace" strategies on the spread of COVID-19.

SEIR-T uses a compartmental formulation that describes viral transmissions and contact tracing in terms of network-defined neighborhoods.

Link to the corresponding paper which documents the model (Andrei et al., 2020): 
https://sotonac-my.sharepoint.com/:b:/g/personal/ar5g15_soton_ac_uk/ESjnbcDA0thLtN4Hr68rCiIBE8q45dcCW_56rsjIavA0iQ?e=5ElCY2

Running example:
    python run.py \
    --netsize 1000 \
    --k 10 \
    --multip 3 \
    --model 'covid' \
    --dual 1 \
    --overlap 1 \
    --uptake .5 \
    --maintain_overlap False \
    --nnets 4 \
    --niters 4 \
    --separate_traced True \
    --noncomp 0 \
    --taut .1 \
    --taur .1 > file.out

Alternatively, call from a Notebook run.run_mock(**kwargs).

Parameter descriptions:
<ul>
<li>'beta': 0.0791, # transmission rate -> For Covid, 0.0791 correponding to R0=3.18; later lockdown estimation: .0806</li>
<li>    'eps': 1/3.7, # latency -> For Covid 3.7 days</li>
<li>    'gamma': 1/2.3, # global (spontaneus) recovey rate -> For Covid 2.3 days</li>
<li>    'spontan': False, # allow spontaneus recovery (for SIR and SEIR only, Covid uses this by default)</li>
<li>    'gammatau': 0, # recovery rate for traced people (if 0, global gamma is used)</li>
<li>    'taur': 0.1, 'taut': 0.1, # random tracing (testing) + contract-tracing rate which will be multiplied with no of traced contacts</li>
<li>    'taut_two': 0.1, # contract-tracing rate for the second tracing network (if exists)</li>
<li>    'noncomp': .02, # noncompliance rate (default: each day the chance of going out of isolation increases by 2%)</li>
<li>    'noncomp_time': True, # whether the noncomp rate will be multiplied by time difference t_current - t_trace</li>
<li>    'noncomp_after': 0, # period after which T becomes automatically N (nonisolating); 0 means disabled; 14 is standard quarantine</li>
<li>    'netsize': 1000, 'k': 10, # net size, avg degree, </li>
<li>    'nettype': 'random', 'p': .05, # network wiring type and a rewire prob for various graph types</li>
<li>    'overlap': .8, 'overlap_two': .4, # overlaps for dual nets (second is used only if dual == 2)</li>
<li>    'zadd': 0, 'zrem': 5, # if no overlap given, these values are used for z_add and z_rem; z_add also informs overlap of additions</li>
<li>    'zadd_two': 0, 'zrem_two': 5, # these are used only if dual == 2 and no overlap_manual is given</li>
<li>    'uptake': 1., 'maintain_overlap': True, </li>
<li>    'nnets': 1, 'niters': 1, 'nevents': 0, # running number of nets, iterations per net and events (if 0, until no more events)</li>
<li>    'multip': 1, # 0 - no multiprocess, 1 - multiprocess nets, 2 - multiprocess iters, 3 - multiprocess nets and iters (half-half cpus)</li>
<li>    'dual': 1, # 0 - tracing happens on same net as infection, 1 - one dual net for tracing, 2 - two dual nets for tracing</li>
<li>    'isolate_s': True, # whether or not Susceptible people are isolated (Note: they will not get infected unless noncompliant)</li>
<li>    'trace_once': False, # if True a node cannot become traced again after being noncompliant</li>
<li>    'draw': False, 'draw_iter': False, # whether to draw at start/finish of simulation or at after each event</li>
<li>    'draw_layout': 'spectral', # networkx drawing layout to use when drawing</li>
<li>    'seed': -1, 'netseed': -1, # seed of infection and exponentials, and the seed for network initializations</li>
<li>    'summary_print': -1, # None -> full_summary never called; False -> no summary printing, True -> print summary as well</li>
<li>    'summary_splits': 1000, # how many time splits to use for the epidemic summary</li>
<li>    'r_window': 7, # number of days for Reff calculation</li>
<li>    'separate_traced': False, # whether to have the Traced state separate from all the other states</li>
<li>    'model': 'sir', # can be sir, seir or covid</li>
<li>    'first_inf': 1., # number of nodes infected at the start of sim</li>
<li>    'rem_orphans': False, # whether or not to remove orphans from the infection network (they will not move state)</li>
<li>    'presample': 0, # number of stateless exponential presamples (if 0, no presampling)</li>
<li>    'earlystop_margin': 0,</li>
<li>    'avg_without_earlystop': False, # whether alternative averages which have no early stopped iterations are to be computed</li>
<li>    'efforts': False,</li>
<li>    # COVID model specific parameters:</li>
<li>    'pa': 0.2, # probability of being asymptomatic (could also be 0.5)</li>
<li>    'rel_beta': .5, # relative infectiousness of Ip/Ia compared to Is (Imperial paper + Medrxiv paper)</li>
<li>    'rel_taur': .8, # relative random tracing (testing) rate of Ia compared to Is </li>
<li>    'miup': 1/1.5, # duration of prodromal phase</li>
<li>    'ph': [0, 0.1, 0.2], # probability of being hospitalized (i.e. having severe symptoms Pss) based on age category </li>
<li>    'lamdahr': [0, .083, .033], # If hospitalized, daily rate entering in R based on age category</li>
<li>    'lamdahd': [0, .0031, .0155], # If hospitalized, daily rate entering in D based on age category</li>
<li>    'group': 1, # Age-group; Can be 0 - children, 1 - adults, 2 - senior</li>
</ul>
