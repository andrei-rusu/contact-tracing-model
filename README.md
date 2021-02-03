# SEIR-T: Model contact tracing for COVID-19

A <b>multi-site mean-field model</b> for studying the effects of different "test and trace" strategies on the spread of <b>COVID-19</b>. The infection spreads over the full network of contacts, whereas <b>each type </b> of tracing gets conducted over different subsets of that network (i.e. one for manual and one for digital tracing).

SEIR-T is a compartmental formulation that describes SARS-Cov-2 transmission and the corresponding contact tracing in terms of network-defined neighborhoods.

Link to original study: Andrei Rusu et al., 2020 https://www.dropbox.com/s/ozdbdlw0351895p/Contact_Tracing.pdf 

<br />

<p style="text-align:center;"><a href="https://www.dropbox.com/s/m284tx8tmgeckqk/noncomp.mp4?dl=0"><img src="fig/noncomp.gif"/></a></p>



<i>Ineffective Manual + Digital Tracing </i>          |  <i>Efficient Manual + Digital Tracing </i>
:-------------------------:|:-------------------------:
[![Ineffective tracing example](fig/nct.gif)](https://www.dropbox.com/s/bxirlp1271teadg/nct.mp4?raw=1#t=0.1) |  [![Efficient tracing example](fig/ct.gif)](https://www.dropbox.com/s/3o0a63ivj8gk82c/ct.mp4?raw=1#t=0.1)

<br />
<h2>Running example:</h2>

```
python run.py \
    --netsize 1000 \
    --k 10 \
    --p .2 \
    --nettype "ws" \
    --multip 3 \
    --model "covid" \
    --dual 2 \
    --uptake .5 \
    --overlap .7 \
    --maintain_overlap False \
    --nnets 4 \
    --niters 4 \
    --separate_traced True \
    --avg_without_earlystop True \
    --trace_once False \
    --first_inf 1 \
    --earlystop_margin 2 \
    --rem_orphans True \
    --noncomp 0 \
    --presample 100000 \
    --pa .2 \
    --taut .1 \
    --taur .1 \
    --sampling_type "min" \
    --netseed 31 \
    --seed 11 > file.out
```

The API supports Python calls from Jupyter Notebooks with the same arguments:
```python
import run
kwargs = {
    'netsize':30, 'nettype':'ws', 'k':5, 'model':'covid', 'dual':2, 
    'uptake':.5, 'overlap':.7, 'taut':.1, 'taur':.1, 'animate':True,
}
run.run_mock(**kwargs)
```

<br />
<h2>Parameter descriptions:</h2>

```python
### Network related parameters:
# net size & avg degree 
'netsize': 1000, 'k': 10,
 # network wiring type and a rewire prob for various graph types
'nettype': 'random', 'p': .05,
# 0 - tracing happens on same net as infection, 1 - one dual net for tracing, 2 - two dual nets for tracing
'dual': 1,
# overlaps for tracing nets (second is used only if dual == 2)
'overlap': .8, 'overlap_two': .4,
# if no overlap given, tracing net has zadd edges added on random, and zrem removed
'zadd': 0, 'zrem': 5,
 # similar as before but for dual == 2 and if no overlap_two given
'zadd_two': 0, 'zrem_two': 5,
# maximum percentage of nodes with at least 1 link (adoption rate)
'uptake': 1.,
# if maintain_overlap True, then the generator will try to accommodate both the uptake and the overlap
'maintain_overlap': False,

### Compartmental model parameters:
# can be sir, seir or covid
'model': 'sir',
 # number of nodes infected at the start of sim
'first_inf': 1.,
# whether to have the Traced status separate from the infection states
'separate_traced': True,
# whether Susceptible people are isolated 
# if True and 'traced', they can't get infected unless noncompliant
'isolate_s': True,
# whether to mark hospitalized as traced
'trace_h': False,
# if True a node cannot become traced again after being noncompliant
'trace_once': False,

### Disease-specific parameters that are common for multiple models:
# transmission rate -> For Covid, 0.0791 correponding to R0=3.18
'beta': 0.0791,
# latency -> For Covid 3.7 days
'eps': 1/3.7,
# global (spontaneus) recovey rate -> For Covid 2.3 days
'gamma': 1/2.3,
 # allow spontaneus recovery (for SIR and SEIR only, for Covid always true)
'spontan': False,
# recovery rate for traced people (if 0, global gamma is used)
'gammatau': 0,

### COVID model specific parameters:
'pa': 0.2, # probability of being asymptomatic (could also be 0.5)
'rel_beta': .5, # relative infectiousness of Ip/Ia compared to Is (Imperial paper + Medrxiv paper)
'rel_taur': .8, # relative random tracing (testing) rate of Ia compared to Is 
'miup': 1/1.5, # duration of prodromal phase
'ph': [0, 0.1, 0.2], # probability of being hospitalized (i.e. having severe symptoms Pss) based on age category 
'lamdahr': [0, .083, .033], # If hospitalized, daily rate entering in R based on age category
'lamdahd': [0, .0031, .0155], # If hospitalized, daily rate entering in D based on age category
'group': 1, # Age-group; Can be 0 - children, 1 - adults, 2 - senior

### Tracing parameters:
# testing (random tracing) rate
'taur': 0.1,
# contact-tracing rates for first & second tracing networks (if exists)
'taut': 0.1,'taut_two': -1.,
# number of days of delay on second tracing network compared to first one
# this is taken into account only if taut_two==-1
'delay_two': 2.,
# noncompliance rate
# each day the chance of going out of isolation increases by x%
'noncomp': .01,
# whether the noncomp rate gets multiplied by time difference t_current - t_trace
'noncomp_time': True,
# period after which T becomes automatically N (nonisolating)
# 0 means disabled; 14 is standard quarantine
'noncomp_after': 0,

### Simulation controlling parameters
# running number of nets, iterations per net and events for each
# nevents == 0, run until no more events
'nnets': 1, 'niters': 1, 'nevents': 0,
 # seed of infection and exponentials, and the seed for network initializations
'seed': -1, 'netseed': -1,
# 0 - no multiprocess, 1 - multiprocess nets, 2 - multiprocess iters, 3 - multiprocess nets and iters (half-half cpus)
'multip': 1,
# dir: the exponential is sampled DIRECTLY from the function registered on the transition object
# each: the transition obj registers only the lambda rates, the exponential is sampled FOR EACH lambda with exp_sampler.py
# min: Gillespie's algorithm; the transition obj registers the lambda rates, ONLY the MINIMUM exponential is sampled based on sum
'sampling_type': 'dir',
# number of stateless exponential presamples (if 0, no presampling)
'presample': 0, 
# whether or not to remove orphans from the infection network (they will not move state)
'rem_orphans': False,
# wehther efforts are to be computed for each type of tracing (random+contact)
'efforts': False,

### Summary-related parameters
# -1 -> full_summary never called; 0 -> no summary printing, 1 -> print summary as well
'summary_print': -1,
# how many time splits to use for the epidemic summary
'summary_splits': 1000,
# number of days for Reff calculation
'r_window': 5, 
# first_inf + earlystop_margin determines if a simulation is regarded as early stopped
'earlystop_margin': 0,
# whether alternative averages which have no early stopped iterations are to be computed
'avg_without_earlystop': False,

### Drawing-related parameters:
# 0 - no draw, 1 - draw at start/finish, 2 - draw and save figure at finish
'draw': 0,
# if not 0 draw after each iteration and sleep for this long
'draw_iter': 0.,
# animates the disease progression, no other info will be printed
'animate': False,
 # networkx drawing layout to use when drawing
'draw_layout': 'spectral',
# whether the legend will contain the full state name or not
'draw_fullname': False,
```