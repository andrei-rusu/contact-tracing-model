import argparse
import random
import numpy as np

from multiprocessing import cpu_count
from multiprocess.pool import Pool

from lib import utils as ut
    
from lib.stats import StatsProcessor
from lib.models import get_transitions_for_model, add_trans
from lib.multirun_engine import EngineNet
from lib.data_utils import DataLoader

# Covid infection-related parameters according to DiDomenico et al. 2020
# https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-020-01698-4#additional-information
PARAMETER_DEFAULTS = {
    ### Network related parameters:
    # network wiring type: this can either be a STR with the name of the random graph
    # or a LIST of predefined networks for infection, tracing or both
    # if its LIST, it can also contain lists of edges to be added dynamically (based on 'update_after' param)
    'nettype': 'random',
    # net size & avg degree & rewire prob for various graph types - these only have EFFECT IF nettype is a known net type given as STR
    'netsize': 1000, 'k': 10, 'p': .05,
    # controls whether edge weights matter and their normalising factor (only has effect when networks with edges have been provided)
    'use_weights': False, 'K_factor': 10,  # if K_factor = 0, no normalization of weights happens
    # 0 - tracing happens on same net as infection, 1 - one dual net for tracing, 2 - two dual nets for tracing
    'dual': 1,
    # overlaps for tracing nets (second is used only if dual == 2)
    'overlap': .8, 'overlap_two': .4,
    # if no overlap given, tracing net has zadd edges added on random, and zrem removed
    'zadd': 0, 'zrem': 5, # if overlap is set, bool(zadd) is still used to infer whether we want to add random edges or only remove
    # similar as before but for dual == 2 and if no overlap_two given
    'zadd_two': 0, 'zrem_two': 5, # zadd_two fully replicates the functionality of zadd for the second dual net
    # maximum percentage of nodes with at least 1 link (adoption rate)
    'uptake': 1., 'uptake_two': 1.,
    # if maintain_overlap True, then the generator will try to accommodate both the uptake and the overlap
    'maintain_overlap': False, 'maintain_overlap_two': True,
    # update network after X days
    'update_after': -1,

    ### Compartmental model parameters:
    # can be sir, seir or covid
    'model': 'sir',
    # number of nodes infected at the start of sim
    'first_inf': 1.,
    # whether to have the Traced status separate from the infection states
    # Note if this is disabled, much of the functionality surrounding noncompliance and self-isolation exit will not work
    'separate_traced': True,
    # whether Susceptible people are isolated 
    # if True and 'traced', they can't get infected unless noncompliant
    'isolate_s': True,
    # whether to mark hospitalized as traced
    'trace_h': False,
    # if -1 a node can become traced again only after exiting self-isolation
    # not -1 a node can become traced again after this amount of time (in days) has elapsed after becoming ilegally N
    'trace_after': 14,

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
    'taut': 0.1, 'taut_two': -1.,
    # number of days of delay on second tracing network compared to first one
    # this is taken into account only if taut_two==-1
    'delay_two': 2.,
    # noncompliance rate; Note this ONLY works for args.separate_traced=True
    # each day the chance of going out of isolation increases by x%
    'noncomp': .01,
    # whether the noncomp rate gets multiplied by time difference t_current - t_trace
    'noncomp_dependtime': True,
    # period after which T becomes automatically N (nonisolating); Note this ONLY works for args.separate_traced=True
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
    # whether or not to remove orphans from the infection network (they will not move state on the infection net)
    'rem_orphans': False,
    # if rem_orphans, noising of links can be calculated either based on the full node list or the active nonorphan list through this param
    'active_based_noising': False,
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
    # if not 0, draw after each iteration and sleep for this long
    'draw_iter': 0.,
    # animates the disease progression, no other info will be printed
    'animate': False,
    # networkx drawing layout to use when drawing
    'draw_layout': 'spring',
    # whether the legend will contain the full state name or not
    'draw_fullname': False,
}

    
def main(args):
    
    ### This block of code is concerned with overwriting in a sensible manner args arguments, based on incomaptibilities and other criteria
    
    # if the special 'socialevol' value was supplied for nettype, the argument will be overwritten with data coming from the Social Evolution dataset
    # this is an easy hook point to supply own logic for loading custom datasets
    if args.nettype == 'socialevol':
        # allow update_after (used to indicate when dynamic edge updates happen) to influence the aggregation of data (either weekly or daily)
        agg = 'W' if args.update_after > 3 else 'D'
        # this effectively reads out the data file and does initial filtering and joining
        loader = DataLoader(agg=agg, proxim_prob=None)
        # supply to the args.nettype a dictionary, with keys being 'nid', 'Wi', 'Wt' (if which_tr>=0), and '0', '1', '2'
        # corresponding to the timestamp of the specific dynamical edge update
        # note that the code also supports custom tracing networks (in the SocialEvol example, choose which_tr>=0 for this scenario to take effect)
        args.nettype = loader.get_edge_data_for_time(which_inf=0, which_tr=None)
            
    # if animation of the infection progress is selected, disable all prints and enable both draw types
    if args.animate:
        ut.block_print()
        if not args.draw: args.draw = 1
        # if no draw_iter selected, set the sleep time between iters to 1
        if not args.draw_iter: args.draw_iter = 1
    else:
        ut.enable_print()
        
    # the following step ensures that unselected seeds are turned to None
    if args.seed == -1: args.seed = None
    if args.netseed == -1: args.netseed = None
        
    
    first_inf_nodes = None
    # reflect in the netsize and the is_dynamic param the fact that we may have supplied a predefined infection network through args.nettype
    if '0' in args.nettype:
        # the graph is dynamic if any update excepting '0' exists
        args.is_dynamic = (len(args.nettype) > 1)
    # else branch is for random graph, which is not dynamic and for which we can use args.netsize to sample the first infected
    else:
        # update number of first infected to reflect absolute values rather than percentage
        # if args.first_inf >= 1 just turn the value into an int and use that as number of first infected
        args.first_inf = int(args.first_inf) if args.first_inf >= 1 else int(args.first_inf * args.netsize)
        # Random first infected across simulations - seed random locally
        first_inf_nodes = random.Random(args.seed).sample(range(args.netsize), args.first_inf)
        # the random graph is not dynamic
        args.is_dynamic = False
    
    # Turn off multiprocessing if only one net and one iteration selected
    if args.nnets == 1 and args.niters == 1: args.multip = 0
    # if multiprocessing selected, stop drawing
    if args.multip: 
        args.animate = False
        args.draw_iter = args.draw = 0
 
    # Set recovery rate for traced people based on whether gammatau was provided
    if not args.gammatau: args.gammatau = args.gamma
    
    # if age-group dependent vars have been provided as array, then choose the value based on inputted age-group 
    if not np.isscalar(args.ph): args.ph = args.ph[args.group]
    if not np.isscalar(args.lamdahr): args.lamdahr = args.lamdahr[args.group]
    if not np.isscalar(args.lamdahd): args.lamdahd = args.lamdahd[args.group]
    
    # Finally, create the object which will hold stats for all simulations
    # Parametrized by args -> will output the parameter configuration used to obtain these results
    stats = StatsProcessor(args)
    
    
    ### The logic of the simulations run follows here:
    # Boolean responsible for determining whether nInfectious = nInfected
    no_exposed = (args.model == 'sir')
    
    # Whether the model is Covid or not
    is_covid = (args.model == 'covid')

    # if exp, then the transition dictionaries will hold functions which also compute the exponentials
    # otherwise, the functionals will only compute the base rates
    exp = (args.sampling_type == 'dir')
    # Transition dictionaries for each network will be populated based on args.model {state->list(state, trans_func)}
    trans_true_items, trans_know_items = get_transitions_for_model(args, exp)

    # populate these variables only if returns_last_net = True
    true_net = know_net = None
    
    # single random tracing and noncompliance rates
    taur = args.taur
    rel_taur = args.rel_taur
    presample = args.presample
    # if no taut_two set, delay_two will be used together with all the given taut rates to compute the taut_two rates 
    set_taut_two = (args.taut_two < 0)
    # we can simulate with a range of tracing rates or with a single one, depending on args.taut supplied
    tracing_rates = np.atleast_1d(args.taut)
    for taut in tracing_rates:
        # tr_rate will be used as the tracing_net.count_importance
        
        print('=========================================================')
        print('Running simulation with parameters:')
        ut.pvar(args.netsize, args.k, args.dual, args.model, owners=False)
        ut.pvar(args.overlap, args.uptake, args.maintain_overlap, owners=False)
        ut.pvar(taut, taur, args.noncomp, args.noncomp_dependtime, owners=False)
        print('=========================================================\n')
        
        # a random tracing rate is needed before any tracing can happen
        if taur > 0:
            # if no explicit taut_two is set, taut_two = 1 / [ 1/taut (days) + delay (days) ]
            if set_taut_two: 
                args.taut_two = (1 / (1 / taut + args.delay_two) if taut != 0 else 0)
            # Tracing for 'S', 'E' happens over know_net depending only on the traced neighbor count of nid (no testing possible)
            # if no contact tracing rate then this transition will not be possible
            tr_func = ut.get_stateful_sampling_func(
                       'expFactorTimesCountImportance', exp=exp, state='T') if (taut or args.taut_two) else None
            add_trans(trans_know_items, 'S', 'T', tr_func)
            add_trans(trans_know_items, 'E', 'T', tr_func)

            # Test and trace functions
            # Tracing for I states which can be found via testing also depend on a random testing rate: args.taur
            tr_and_test_func = ut.get_stateful_sampling_func(
                       'expFactorTimesCountImportance', exp=exp, state='T', base=taur)
            # For certain states, random tracing is done at a smaller rate (Ia vs Is)
            tr_and_test_rel_func = ut.get_stateful_sampling_func(
                       'expFactorTimesCountImportance', exp=exp, state='T', base=(taur*rel_taur))

            # Update transition parameters based on the abvoe defined tracing functions
            if is_covid:
                # We assume 'I(p)' will not be spotted via testing (false negatives in the first week)
                add_trans(trans_know_items, 'I', 'T', tr_func)
                # Tracing for 'Ia' and 'Is' also depends on a random tracing rate (due to random testing)
                add_trans(trans_know_items, 'Is', 'T', tr_and_test_func)
                # Asymptomatics have a relative testing rate (lower number of asymptomatics actually get tested)
                add_trans(trans_know_items, 'Ia', 'T', tr_and_test_rel_func)
            else:
                # in non-COVID models we assume all 'I' states can be spotted via testing
                add_trans(trans_know_items, 'I', 'T', tr_and_test_func)
        
        # if the tracing events are not separate from the infection events, then allow for traced individuals to get Removed
        if not args.separate_traced:
            # Recovery for traced nodes is network independent at rate gammatau
            add_trans(trans_know_items, 'T', 'R', ut.get_stateless_sampling_func(args.gammatau, exp))
        
        
        nnets = args.nnets
        net_range = range(nnets)
        # Multiprocessing object to use for each network initialization
        engine = EngineNet(args=args, first_inf_nodes=first_inf_nodes, no_exposed=no_exposed, is_covid=is_covid,
                           tr_rate=taut, trans_true_items=trans_true_items, trans_know_items=trans_know_items)

        if args.multip == 1:
            with Pool() as pool:
                for inet, net_events in enumerate(ut.tqdm_redirect(pool.imap(engine, net_range), total=nnets,
                                                                desc='Networks simulation progress')):
                    # Record sim results
                    stats.sim_summary[inet] = net_events
                    
        elif args.multip == 3:
            # allocate half cpus to joblib for parallelizing simulations for different network initializations
            jobs = int(cpu_count() / 2)
            with ut.NoDaemonPool(jobs) as pool:
                for inet, net_events in enumerate(ut.tqdm_redirect(pool.imap(engine, net_range), total=nnets,
                                                                desc='Networks simulation progress')):
                    # Record sim results
                    stats.sim_summary[inet] = net_events
                    
#             with ut.tqdm_joblib(tqdm(desc='Networks simulation progress', file=stdout, total=nnets)), \
#                 Parallel(n_jobs=jobs) as parallel:
#                     all_events = parallel(delayed(engine)(inet) for inet in net_range)
#             stats.sim_summary.update(enumerate(all_events))

        else:
            for inet in net_range:
                print('----- Simulating network no.', inet, '-----')

                # Run simulation
                net_events, true_net, know_net = engine(inet, return_last_net=True)

                # Record sim results
                stats.sim_summary[inet] = net_events

                    
        stats.results_for_param(taut)
        
    if args.summary_print != -1:
        return stats.full_summary(args.summary_splits, args.summary_print, args.r_window)

    return stats, true_net, know_net

            
def run_mock(**kwargs):
    """
    Mocks running simulations from the command line, but offers a coding interface to Notebooks
    """
    
    argmock = argparse.Namespace()

    for k in PARAMETER_DEFAULTS.keys():
        vars(argmock)[k] = kwargs.get(k, PARAMETER_DEFAULTS[k])
    
    return main(argmock)
    
    
if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    
    for k in PARAMETER_DEFAULTS.keys():
        default = PARAMETER_DEFAULTS[k]
        if k == 'taut':
            argparser.add_argument('--' + k, type=float, nargs="+", default=default)
        else:
            # The following is needed since weirdly bool('False') = True in Python
            typed = type(default) if type(default) != bool else lambda arg: arg.lower() in ("yes", "true", "t", "1")
            argparser.add_argument('--' + k, type=typed, default=default)

    args = argparser.parse_args()
    args.summary_print = 1 # If script run, full_summary in print mode will always be called

    main(args)