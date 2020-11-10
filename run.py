import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from tqdm import tqdm
from time import sleep
from itertools import count
from collections import defaultdict
from pathos.pools import ProcessPool as Pool
from joblib import Parallel, delayed
from sys import stdout

from lib import network
from lib.utils import tqdm_redirect, tqdm_joblib, ListDelegator, \
    expFactorTimesCount, expFactorTimesTimeDif, expFactorTimesCountImportance
    
from lib.simulation import Simulation
from lib.stats import StatsProcessor, StatsEvent
from lib.models import get_transitions_for_model, add_trans

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
    'noncomp': .002, # noncompliance rate
    'noncomp_time': True, # whether the noncomp rate will be multiplied by time diference between tracing and current time
    'netsize': 1000, 'k': 10, # net size and avg degree
    'overlap': .8, 'overlap_two': .4, # overlaps for dual nets (second is used only if dual == 2)
    'zadd': 0, 'zrem': 5, # if no overlap given, these values are used for z_add and z_rem; z_add also informs overlap of additions
    'zadd_two': 0, 'zrem_two': 5, # these are used only if dual == 2 and no overlap_manual is given
    'uptake': 1., 'maintain_overlap': True, 
    'nnets': 1, 'niters': 1, 'nevents': 0, # running number of nets, iterations per net and events (if 0, until no more events)
    'multip': 1, # 0 - no multiprocess, 1 - multiprocess nets, 2 - multiprocess iters, 3 - multiprocess nets and iters (half-half cpus)
    'draw': False, 'draw_iter': False, 'dual': 1, 'seed': None,
    # None -> full_summary never called; False -> no summary printing, True -> print summary as well
    'summary_print': None,
    'summary_splits': 1000, # how many time splits to use for the epidemic summary
    'separate_traced': False, # whether to have the Traced state separate from all the other states
    'model': 'sir', # can be sir, seir or covid
    'first_inf': 1,
    'rem_orphans': False,
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
    
    # Random first infected across simulations
    first_inf = random.sample(range(args.netsize), args.first_inf)
    
    # Boolean responsible for determining whether nInfectious = nInfected
    no_exposed = (args.model == 'sir')
    
    # Whether the model is Covid or not
    is_covid = (args.model == 'covid')
    
    # Turn off multiprocessing if only one net and one iteration selected
    if args.nnets == 1 and args.niters == 1: args.multip = False
 
    # Set recovery rate for traced people based on whether gammatau was provided
    if args.gammatau is None: args.gammatau = args.gamma
    
    # if age-group dependent vars have been provided as array, then choose the value based on inputted age-group 
    if not np.isscalar(args.ph): args.ph = args.ph[args.group]
    if not np.isscalar(args.lamdahr): args.lamdahr = args.lamdahr[args.group]
    if not np.isscalar(args.lamdahd): args.lamdahd = args.lamdahd[args.group]
    
    # Transition dictionaries for each network will be populated based on args.model {state->list(state, trans_func)}
    trans_true_items, trans_know_items = get_transitions_for_model(args)
    # if no noncompliace rate is chosen, skip this transition
    # Note: optional argument time can be passed to make the noncompliance rate time dependent
    if args.noncomp:
        if args.noncomp_time:
            noncomp_func = lambda net, nid, time: expFactorTimesTimeDif(net, nid, current_time=time, lamda=args.noncomp)
        else:
            noncomp_func = lambda net, nid, time=None: -(math.log(random.random()) / args.noncomp)
        add_trans(trans_know_items, 'T', 'N', noncomp_func)
    
    # we can simulate with a range of tracing rates or with a single one provied by args.taut
    tracing_rates = np.atleast_1d(args.taut)
    
    # populate these variables only if returns_last_net = True
    true_net = know_net = None
    
    for tr_rate in tracing_rates:
        # tr_rate will be used as the tracing_net.count_importance
        
        print('===== For taut =', tr_rate, '& taur =', args.taur, "=====")

        # Tracing for 'S', 'E' happens over know_net depending only on the traced neighbor count of nid (no testing possible)
        tr_func = (lambda net, nid: expFactorTimesCountImportance(net, nid, state='T', base=0)) if tr_rate else None
        add_trans(trans_know_items, 'S', 'T', tr_func)
        add_trans(trans_know_items, 'E', 'T', tr_func)
        
        # Test and trace functions
        tr_and_test_func = None
        tr_and_test_rel_func = None
        if tr_rate or args.taur:
            # Tracing for I states which can be found via testing also depend on a random testing rate: args.taur
            tr_and_test_func = \
                lambda net, nid: expFactorTimesCountImportance(net, nid, state='T', base=args.taur)
            # For certain states, random tracing is done at a smaller rate (Ia vs Is)
            tr_and_test_rel_func = \
                lambda net, nid: expFactorTimesCountImportance(net, nid, state='T', base=args.taur * args.rel_taur)
        
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
        engine = EngineNet(args=args, first_inf=first_inf, no_exposed=no_exposed, is_covid=is_covid,
                          tr_rate=tr_rate, trans_true_items=trans_true_items, trans_know_items=trans_know_items)

        if args.multip == 1:
            with Pool() as pool:
                for inet, net_events in enumerate(tqdm_redirect(pool.imap(engine, net_range), total=args.nnets, 
                                                                desc='Networks simulation progress')):
                    # Record sim results
                    stats.sim_summary[inet] = net_events
                    
        elif args.multip == 3:
            # allocate half cpus to joblib for parallelizing simulations for different network initializations
            jobs = int(multiprocessing.cpu_count() / 2)
            with tqdm_joblib(tqdm(desc='Networks simulation progress', file=stdout, total=nnets)), Parallel(n_jobs=jobs) as parallel:
                all_events = parallel(delayed(engine)(inet) for inet in net_range)
            stats.sim_summary.update(enumerate(all_events))
            
        else:
            for inet in net_range:
                print('----- Simulating network no.', inet, '-----')

                # Run simulation
                net_events, true_net, know_net = engine(inet, return_last_net=True)

                # Record sim results
                stats.sim_summary[inet] = net_events

                    
        stats.results_for_param(tr_rate)
        
    if args.summary_print is not None:
        return stats.full_summary(args.summary_splits, args.summary_print)

    return stats, true_net, know_net


# Classes used to perform Multiprocessing iterations

class Engine():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def reinit_net(self, first_inf):
        self.true_net.init_states('S')
        self.true_net.change_state(first_inf, state='I', update=True)
        
    def __call__(self, itr):
        raise NotImplementedError
        
        
class EngineNet(Engine):
    """
    This class will be used for multiprocessing over different network initialization
    """
    
    def __call__(self, inet, return_last_net=False):
        # local vars for efficiency
        args = self.args
        first_inf = self.first_inf
        no_exposed = self.no_exposed
        is_covid = self.is_covid
        tr_rate = self.tr_rate
        trans_true_items = self.trans_true_items
        trans_know_items = self.trans_know_items
        
        # will hold all network events
        net_events = defaultdict()
        
        # Get true_net with all nodes in state 'S' but one which is 'I'
        true_net = network.get_random(args.netsize, args.k, args.rem_orphans)
        true_net.change_state(first_inf, state='I', update=True)

        # Placeholder for the dual network (will be initialized only if args.dual)
        know_net = None

        if args.dual:
            # First dual net depends on both overlap and uptake (this is usually the digital contact tracing net)
            know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem, 
                            keep_nodes_percent=args.uptake, maintain_overlap=args.maintain_overlap, count_importance=tr_rate)

            # if 2 dual networks selected, create the second network and add both to a ListDelegator
            if args.dual == 2:
                # Second net depends only on overlap_two - i.e. uptake = 1 (this is usually the manual tracing net)
                know_net_two = network.get_dual(true_net, args.overlap_two, args.zadd_two, args.zrem_two, 
                            keep_nodes_percent=1, maintain_overlap=True, count_importance=args.taut_two)

                # know_net becomes a ListDelegator of the 2 networks
                know_net = ListDelegator(know_net, know_net_two)


            # Object used during Multiprocessing of Network simulation events
            engine = EngineDual(
                args=args, no_exposed=no_exposed, is_covid=is_covid,
                true_net=true_net, know_net=know_net, tr_rate=tr_rate,
                trans_true=trans_true_items, trans_know=trans_know_items
            )

        else:
            # Object used during Multiprocessing of Network simulation events
            engine = EngineOne(
                args=args, no_exposed=no_exposed, is_covid=is_covid,
                true_net=true_net, tr_rate=tr_rate,
                trans_true=trans_true_items,
            )


        niters = args.niters
        iters_range = range(niters)

        if args.multip == 2 or args.multip == 3:
            # allocate EITHER half or all cpus to pathos.multiprocess for parallelizing simulations for different iterations of 1 init
            # multip == 2 parallelize only iterations; multip == 3 parallelize both net and iters
            jobs = int(multiprocessing.cpu_count() / (args.multip - 1))
            with Pool(ncpus=jobs) as pool:
                for itr, stats_events in enumerate(tqdm_redirect(pool.imap(engine, iters_range), total=niters,
                                                                desc='Iterations simulation progress')):
                    # Record sim results
                    net_events[itr] = stats_events
            
        else:
            for itr in tqdm_redirect(iters_range, desc='Iterations simulation progress'):
                print('Running iteration ' + str(itr) + ':')
                
                # Reinitialize network + Random first infected at the beginning of each run BUT the first one
                # This is needed only in sequential processing since in multiprocessing the nets are deepcopied anyway
                if itr:
                    engine.reinit_net(first_inf)

                # Run simulation
                stats_events = engine(itr)

                # Record sim results
                net_events[itr] = stats_events
                print('---> Result:' + str(stats_events[-1]['totalInfected']) + ' total infected persons over time.')
                
        if return_last_net:
            return net_events, true_net, know_net
        
        return net_events

    
class EngineDual(Engine):
    
    def reinit_net(self, first_inf):    
        super().reinit_net(first_inf)
        self.know_net.init_states('S')
        self.know_net.change_state(first_inf, state='I', update=True)
        
    def __call__(self, itr):
        
        # local vars for efficiency
        args = self.args
        true_net = self.true_net
        # Note: know_net may be a ListDelegator of tracing networks
        know_net = self.know_net

        # Draw network if flag
        if args.draw:
            # we make plots for the true network + all the dual networks
            fix, ax = plt.subplots(nrows=1, ncols=int(args.dual) + 1, figsize=(16, 5))
            ax[0].set_title('True Network')
            self.true_net.draw(seed=args.seed, show=False, ax=ax[0])
            ax[1].set_title('Tracing Networks')
            self.know_net.draw(pos=true_net.pos, show=False, ax=ax[1:])
            plt.show()

        # simulation objects
        sim_true = true_net.get_simulator(self.trans_true)
        sim_know = know_net.get_simulator(self.trans_know)
        
        # number of initial infected
        inf = args.first_inf

        # metrics to record simulation summary
        m = {
            'nI' : inf,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : inf,
            'totalInfectious': inf,
            'totalFalsePositive': 0,
            'totalTraced' : 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered' : 0,
            'totalHospital' : 0,
            'totalDeath' : 0,
            'tracingEffortRandom' : -1,
            'tracingEffortContact' : -1,
        }
        
        # results over time recorded
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run exactly args.nevents otherwise
        events_range = range(args.nevents) if args.nevents else count()
        for i in events_range:
            
            e1 = sim_true.get_next_event()
            
            if args.separate_traced:
                # get events on the known network ONLY IF a tracing rate actually exists
                e2 = sim_know.get_next_trace_event() if self.tr_rate or args.taur else None

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break
                
                # get all event candidates, assign 'inf' to all Nones, and select the event with the smallest 'time' value
                candidates = np.append(np.atleast_1d(e2), e1)
                times = [e_candidate.time if e_candidate is not None else float('inf') for e_candidate in candidates]
                e = candidates[np.argmin(times)]
                                
                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                if e.to == 'T':
                    # the update to total traced counts is done only once (ignore if same nid is traced again)
                    if e.node not in true_net.traced_time:
                        m['totalTraced'] += 1
                        # if S -> T then a person has been incorrectly traced
                        if e.fr == 'S': m['totalFalseTraced'] += 1
                        if e.fr == 'E': m['totalExposedTraced'] += 1
                    
                    sim_true.run_trace_event(e, True)
                    sim_know.run_trace_event(e, True)
                    
                # Non-compliance with isolation event
                elif e.fr == 'T':
                    # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                    if e.node not in true_net.noncomp_time:
                        m['totalNonCompliant'] += 1
                        
                    sim_true.run_trace_event(e, False)
                    sim_know.run_trace_event(e, False)
                    
                # otherwise, normal logic follows (with update counts)
                else:
                    sim_true.run_event(e)
                    sim_know.run_event(e)
                    # Update event.FROM counts:
                    #   - for models other than covid, leaving 'I' means a decrease in infectious count
                    #   - for covid, leaving 'Ia' or 'Is' means current infectious count decreases
                    if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                        m['nI'] -= 1
                    elif e.fr == 'E':
                        m['nE'] -= 1
                    elif e.fr == 'H':
                        m['nH'] -= 1
            
            else:
                # get events on the known network ONLY IF a tracing rate actually exists
                e2 = sim_know.get_next_event() if self.tr_rate or args.taur else None

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break
                    
                # get all event candidates, assign 'inf' to all Nones, and select the event with the smallest 'time' value
                candidates = np.append(np.atleast_1d(e2), e1)
                times = [e_candidate.time if e_candidate is not None else float('inf') for e_candidate in candidates]
                e = candidates[np.argmin(times)]

                sim_true.run_event(e)
                sim_know.run_event(e)
                
                # Update event.FROM counts:
                #   - for models other than covid, leaving 'I' means a decrease in infectious count
                #   - for covid, leaving 'Ia' or 'Is' means current infectious count decreases
                if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                    m['nI'] -= 1
                elif e.fr == 'E':
                    m['nE'] -= 1
                elif e.fr == 'H':
                    m['nH'] -= 1
                    
                # Update 'T' counts if a tracing event was chosen
                # Note that this could double count if T -> N -> T multiple times
                if e.to == 'T':
                    m['totalTraced'] += 1
                    # if S -> T then a person has been incorrectly traced
                    if e.fr == 'S': m['totalFalseTraced'] += 1
                    if e.fr == 'E': m['totalExposedTraced'] += 1
            
            
            # event.TO Updates are common for all models and all parameter settings
            
            if e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if self.no_exposed: m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRecovered'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1
                    
            # list of efforts and final state check for each tracing network
            efforts_and_check_end = np.atleast_2d(know_net.compute_efforts_and_check_end_config())
            # random (testing) effort will be the same for all dual networks - element 0 in each result list      
            m['tracingEffortRandom'] = round(args.taur * efforts_and_check_end[0, 0], 2)
            # the contact tracing effort is DIFFERENT across the dual networks - element 1 in each result list
            m['tracingEffortContact'] = self.tr_rate * efforts_and_check_end[:, 1]
            # the flag to check for sim finish is also the same for all networks - last element in each result list
            stop_simulation = efforts_and_check_end[0, -1]
            
            # False Positives: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalsePositive'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))

            # draw network at each inner state if option selected
#             if args.draw_iter:
#                 print('State after events iteration ' + str(i) + ':')
#                 fix, ax = plt.subplots(nrows=1, ncols=int(args.dual) + 1, figsize=(16, 5))
#                 ax[0].set_title('True Network')
#                 self.true_net.draw(show=False, ax=ax[0])
#                 ax[1].set_title('Tracing Networks')
#                 self.know_net.draw(pos=self.true_net.pos, show=False, ax=ax[1:])
#                 plt.show()
#                 sleep(1.5)

            # close simulation if the only possible events remain T -> N -> T etc
            if stop_simulation:
                break

        if args.draw:
            print('Final state:')
            fix, ax = plt.subplots(nrows=1, ncols=int(args.dual) + 1, figsize=(16, 5))
            ax[0].set_title('True Network')
            self.true_net.draw(show=False, ax=ax[0])
            ax[1].set_title('Tracing Networks')
            self.know_net.draw(pos=self.true_net.pos, show=False, ax=ax[1:])
            plt.show()               

        return result

            
class EngineOne(Engine):
    
    def __call__(self, itr):
        
        # local vars for efficiency
        args = self.args
        true_net = self.true_net
        trans = self.trans_true
        
        if self.args.draw:
            true_net.draw(seed=args.seed)
            
        
        # in the case of separate_traced, mimic dual simulation objects as if dual net was run (sim_true + sim_know)
        if args.separate_traced:
            # infection simulator in this case should not be able to run 'T' events
            transition_items = defaultdict(list)
            for state in trans:
                for transition, func in trans[state]:
                    if transition != 'T':
                        transition_items[state].append((transition, func))
            sim_true = true_net.get_simulator(transition_items)
            # the tracing simulator will callibrate on the first call to only accept 'T' events
            sim_know = true_net.get_simulator(trans)
        else:
            # base simulation object with all transitions possible
            sim_true = true_net.get_simulator(trans)
        
        # number of initial infected
        inf = self.args.first_inf

        # metrics to record simulation summary
        m = {
            'nI' : inf,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : inf,
            'totalInfectious': inf,
            'totalFalsePositive': 0,
            'totalTraced' : 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered' : 0,
            'totalHospital' : 0,
            'totalDeath' : 0,
            'tracingEffortRandom' : -1,
            'tracingEffortContact' : -1,
        }

        # will hold up all results across time
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run args.nevents otherwise
        events_range = range(self.args.nevents) if self.args.nevents else count()

        for i in events_range:

            e1 = sim_true.get_next_event()
            
            ### NOTE: Running separate_traced sims on a Single network is actually SLOWER than running on Dual net directly
            if self.args.separate_traced:
                # allow for separate tracing events to also be considered in the current events loop
                e2 = sim_know.get_next_trace_event()

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2
                
                # update the time for the 'fake' sim_know object
                sim_know.time = e.time

                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                if e.to == 'T':
                    # the update to total traced counts is done only once (ignore if same nid is traced again)
                    if e.node not in true_net.traced_time:
                        m['totalTraced'] += 1
                        # if S -> T then a person has been incorrectly traced
                        if e.fr == 'S': m['totalFalseTraced'] += 1
                        if e.fr == 'E': m['totalExposedTraced'] += 1
                    
                    sim_true.run_trace_event(e, True)
                    
                # Non-compliance with isolation event
                elif e.fr == 'T':
                    # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                    if e.node not in true_net.noncomp_time:
                        m['totalNonCompliant'] += 1
                        
                    sim_true.run_trace_event(e, False)
                    
                else:
                    sim_true.run_event(e)

                    # Update event.FROM counts:
                    #   - for models other than covid, leaving 'I' means a decrease in infectious count
                    #   - for covid, leaving 'Ia' or 'Is' means current infectious count decreases
                    if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                        m['nI'] -= 1
                    elif e.fr == 'E':
                        m['nE'] -= 1
                    elif e.fr == 'H':
                        m['nH'] -= 1
                    
            else:
                # If no more events left, break out of the loop
                if e1 is None:
                    break
                
                e = e1

                sim_true.run_event(e)

                # for models other than covid, leaving 'I' means a decrease in infectious
                # for covid, leaving Ia or Is state means current infectious decreases
                if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                    m['nI'] -= 1
                elif e.fr == 'E':
                    m['nE'] -= 1
                elif e.fr == 'H':
                    m['nH'] -= 1
                    
                # Update 'T' fr counts
                if e.to == 'T':
                    m['totalTraced'] += 1
                    # if S -> T then a person has been incorrectly traced
                    if e.fr == 'S': m['totalFalseTraced'] += 1
                    if e.fr == 'E': m['totalExposedTraced'] += 1
                    
            
            # event.TO Updates are common for all models
                                
            if e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if self.no_exposed: m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRecovered'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1

                
            # list of efforts and final state check for each tracing network (in this case only one)
            efforts_and_check_end = true_net.compute_efforts_and_check_end_config()
            # random (testing) effort will be the same for all dual networks - element 0 in the result list      
            m['tracingEffortRandom'] = round(args.taur * efforts_and_check_end[0], 2)
            # the contact tracing effort is DIFFERENT across the dual networks - element 1 in the result list
            m['tracingEffortContact'] = round(self.tr_rate * efforts_and_check_end[1], 2)
            # the flag to check for sim finish is also the same for all networks - last element in the result list
            stop_simulation = efforts_and_check_end[-1]
            
            # False Positives: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalsePositive'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))
            
#             if self.args.draw_iter:
#                 print('State after events iteration ' + str(i) + ':')
#                 self.true_net.draw(seed=args.seed)

            # close simulation if the only possible events remain T -> N -> T etc
            if stop_simulation:
                break
            
        if self.args.draw:
            print('Final state:')
            self.true_net.draw()
            
        return result


            
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
    args.summary_print = True # If script run, full_summary in print mode will always be called

    main(args)