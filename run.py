import argparse
import random
import numpy as np
import multiprocessing

from sys import stdout
from tqdm import tqdm
from multiprocessing import cpu_count
from multiprocess.pool import Pool
from joblib import Parallel, delayed

from lib import utils as ut
    
from lib.stats import StatsProcessor
from lib.models import get_transitions_for_model, add_trans
from lib.multirun_engine import EngineNet
from lib.exp_sampler import ExpSampler

# Covid default parameter values according to Domenico et al. 2020
# https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-020-01698-4#additional-information
PARAMETER_DEFAULTS = {
    'beta': 0.0791, # transmission rate -> For Covid, 0.0791 correponding to R0=3.18; later lockdown estimation: .0806
    'eps': 1/3.7, # latency -> For Covid 3.7 days
    'gamma': 1/2.3, # global (spontaneus) recovey rate -> For Covid 2.3 days
    'spontan': False, # allow spontaneus recovery (for SIR and SEIR only, Covid uses this by default)
    'gammatau': None, # recovery rate for traced people (if None, global gamma is used)
    'taur': 0.1, 'taut': 0.1, # random tracing (testing) + contract-tracing rate which will be multiplied with no of traced contacts
    'taut_two': 0.1, # contract-tracing rate for the second tracing network (if exists)
    'noncomp': .02, # noncompliance rate
    'noncomp_time': True, # whether the noncomp rate will be multiplied by time diference between tracing and current time
    'netsize': 1000, 'k': 10, # net size and avg degree
    'overlap': .8, 'overlap_two': .4, # overlaps for dual nets (second is used only if dual == 2)
    'zadd': 0, 'zrem': 5, # if no overlap given, these values are used for z_add and z_rem; z_add also informs overlap of additions
    'zadd_two': 0, 'zrem_two': 5, # these are used only if dual == 2 and no overlap_manual is given
    'uptake': 1., 'maintain_overlap': True, 
    'nnets': 1, 'niters': 1, 'nevents': 0, # running number of nets, iterations per net and events (if 0, until no more events)
    'multip': 1, # 0 - no multiprocess, 1 - multiprocess nets, 2 - multiprocess iters, 3 - multiprocess nets and iters (half-half cpus)
    'dual': 1, # 0 - tracing happens on same net as infection, 1 - one dual net for tracing, 2 - two dual nets for tracing
    'isolate_s': True, # whether or not Susceptible people are isolated (Note: they will not get infected unless noncompliant)
    'draw': False, 'draw_iter': False, 'seed': None,
    'summary_print': -1, # None -> full_summary never called; False -> no summary printing, True -> print summary as well
    'summary_splits': 1000, # how many time splits to use for the epidemic summary
    'separate_traced': False, # whether to have the Traced state separate from all the other states
    'model': 'sir', # can be sir, seir or covid
    'first_inf': 1, # number of nodes infected at the start of sim
    'rem_orphans': False, # whether or not to remove orphans from the infection network (they will not move state)
    'presample': 0, # number of stateless exponential presamples (if 0, no presampling)
    'earlystop_margin': 0,
    'avg_without_earlystop': False, # whether alternative averages which have no early stopped iterations are to be computed
    
    # COVID model specific parameters:
    'pa': 0.2, # probability of being asymptomatic (could also be 0.5)
    'rel_beta': .5, # relative infectiousness of Ip/Ia compared to Is (Imperial paper + Medrxiv paper)
    'rel_taur': .8, # relative random tracing (testing) rate of Ia compared to Is 
    'miup': 1/1.5, # duration of prodromal phase
    'ph': [0, 0.1, 0.2], # probability of being hospitalized (i.e. having severe symptoms Pss) based on age category 
    'lamdahr': [0, .083, .033], # If hospitalized, daily rate entering in R based on age category
    'lamdahd': [0, .0031, .0155], # If hospitalized, daily rate entering in D based on age category
    'group': 1, # Age-group; Can be 0 - children, 1 - adults, 2 - senior
}

    
def main(args):
    
    # Will hold stats for all simulations
    stats = StatsProcessor(args)
    
    # seed the random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Random first infected across simulations
    first_inf_nodes = random.sample(range(args.netsize), args.first_inf)
    
    # Boolean responsible for determining whether nInfectious = nInfected
    no_exposed = (args.model == 'sir')
    
    # Whether the model is Covid or not
    is_covid = (args.model == 'covid')
    
    # Turn off multiprocessing if only one net and one iteration selected
    if args.nnets == 1 and args.niters == 1: args.multip = 0
    # if multiprocessing selected, stop drawing
    if args.multip: args.draw = args.draw_iter = False
 
    # Set recovery rate for traced people based on whether gammatau was provided
    if args.gammatau is None: args.gammatau = args.gamma
    
    # if age-group dependent vars have been provided as array, then choose the value based on inputted age-group 
    if not np.isscalar(args.ph): args.ph = args.ph[args.group]
    if not np.isscalar(args.lamdahr): args.lamdahr = args.lamdahr[args.group]
    if not np.isscalar(args.lamdahd): args.lamdahd = args.lamdahd[args.group]
        
    # single random tracing and noncompliance rates
    taur = args.taur
    rel_taur = args.rel_taur
    noncomp = args.noncomp
    # number of presamples stateless exponentials per each base rate
    presample = args.presample

    # Transition dictionaries for each network will be populated based on args.model {state->list(state, trans_func)}
    trans_true_items, trans_know_items = get_transitions_for_model(args)
    # if no noncompliace rate is chosen, skip this transition
    # Note: optional argument time can be passed to make the noncompliance rate time dependent
    if noncomp:
        if args.noncomp_time:
            noncomp_func = lambda net, nid, time: ut.expFactorTimesTimeDif(net, nid, current_time=time, lamda=noncomp)
        else:
            noncomp_func = ut.get_stateless_exp_sampler(noncomp, presample)
        add_trans(trans_know_items, 'T', 'N', noncomp_func)

    # populate these variables only if returns_last_net = True
    true_net = know_net = None
    
    # we can simulate with a range of tracing rates or with a single one, depending on args.taut supplied
    tracing_rates = np.atleast_1d(args.taut)
    for taut in tracing_rates:
        # tr_rate will be used as the tracing_net.count_importance
        
        print('=========================================================')
        print('Running simulation with parameters:')
        ut.pvar(args.netsize, args.k, args.dual, args.model, owners=False)
        ut.pvar(args.overlap, args.uptake, args.maintain_overlap, owners=False)
        ut.pvar(taut, taur, noncomp, args.noncomp_time, owners=False)
        print('=========================================================\n')
        
        # a random tracing rate is needed before any tracing can happen
        if taur > 0:
            # Tracing for 'S', 'E' happens over know_net depending only on the traced neighbor count of nid (no testing possible)
            # if no contact tracing rate then this transition will not be possible
            tr_func = (lambda net, nid: ut.expFactorTimesCountImportance(net, nid, state='T', base=0)) if taut else None
            add_trans(trans_know_items, 'S', 'T', tr_func)
            add_trans(trans_know_items, 'E', 'T', tr_func)

            # Test and trace functions
            # Tracing for I states which can be found via testing also depend on a random testing rate: args.taur
            tr_and_test_func = \
                lambda net, nid: ut.expFactorTimesCountImportance(net, nid, state='T', base=taur)
            # For certain states, random tracing is done at a smaller rate (Ia vs Is)
            tr_and_test_rel_func = \
                lambda net, nid: ut.expFactorTimesCountImportance(net, nid, state='T', base=taur * rel_taur)

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
        return stats.full_summary(args.summary_splits, args.summary_print)

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
        # The following is needed since weirdly bool('False') = True in Python
        typed = type(default) if type(default) != bool else lambda arg: arg.lower() in ("yes", "true", "t", "1")
        argparser.add_argument('--' + k, type=typed, default=default)

    args = argparser.parse_args()
    args.summary_print = 1 # If script run, full_summary in print mode will always be called

    main(args)