import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import math

from tqdm import tqdm
from time import sleep
from itertools import count
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool as Pool

from lib import network
from lib.utils import tqdm_redirect, expFactorTimesCount, expFactorTimesTimeDif
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
    'noncomp': .002,
    'netsize': 1000, 'k': 10, # net size and avg degree
    'overlap': .08, 'zadd': 0, 'zrem': 5, # net overlap translated to z_rem / z_add & z_rem OR specific z_add, z_rem if overlap is None
    'nnets': 1, 'niters': 1, 'nevents': 0, # running number of nets, iterations per net and events (if 0, until no more events)
    'multip': False, # whether multiprocessing is used
    'draw': False, 'draw_iter': False, 'dual': True, 'seed': 43,
    # None -> full_summary never called; False -> no summary printing, True -> print summary as well
    'summary_print': None,
    'summary_splits': 1000, # how many time splits to use for the epidemic summary
    'separate_traced': False, # whether to have the Traced state separate from all the other states
    'model': 'sir', # can be sir, seir or covid
    'first_inf': 1,
    # COVID model specific parameters:
    'pa': 0.2, # probability of being asymptomatic (could also be 0.5)
    'rel_beta': .5, # relative infectiousness of Ip/Ia compared to Is (Imperial paper + Medrxiv paper)
    'rel_taur': .8, # relative random tracing (testing) rate of Ia compared to Is 
    'miup': 1/1.5, # duration of prodromal phase
    'ph': [0, 0.1, 0.2], # probability of being hospitalized (i.e. having severe symptoms Pss) based on age category 
    'lamdahr': [0, .083, .033], # If hospitalized, daily rate entering in R based on age category
    'lamdahd': [0, .0031, .0155], # If hospitalized, daily rate entering in D based on age category
    'group': 1, # Age-group; Can be 0 - children, 1 - adults, 2 - senior
    'rem_orphans': False,
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
        add_trans(trans_know_items, 'T', 'N', lambda net, nid, time=None: expFactorTimesTimeDif(net, nid, args.noncomp, time))
    
    # we can simulate with a range of tracing rates or with a single one provied by args.taut
    tracing_rates = np.atleast_1d(args.taut)
    
    for tr_rate in tracing_rates:
        
        print('For taut =', tr_rate)

        # Tracing for 'S', 'E' happens over know_net depending only on the traced neighbor count of nid (no testing possible)
        tr_func = (lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=0)) if tr_rate else None
        add_trans(trans_know_items, 'S', 'T', tr_func)
        add_trans(trans_know_items, 'E', 'T', tr_func)
        
        # Test and trace functions
        tr_and_test_func = None
        tr_and_test_rel_func = None
        if tr_rate or args.taur:
            # Tracing for I states which can be found via testing also depend on a random testing rate: args.taur
            tr_and_test_func = \
                lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=args.taur)
            # For certain states, random tracing is done at a smaller rate (Ia vs Is)
            tr_and_test_rel_func = \
                lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=args.taur * args.rel_taur)
        
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
                
                
        for inet in range(args.nnets):
            print('Simulating network - No.', inet)
            
            # Get true_net with all nodes in state 'S' but one which is 'I'
            true_net = network.get_random(args.netsize, args.k, args.rem_orphans)
            true_net.change_state(first_inf, state='I', update=True)
            
            # Placeholder for the dual network (will be initialized only if args.dual)
            know_net = None
                        
            if args.dual:
                # Priority will be given to the overlap value
                # Normally z_add = z_rem = z, but if z_add = 0 then only z_rem=z is produced
                # If overlap is None, zadd and zrem values are used
                know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem)
                
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

            iters_range = range(args.niters)

            if args.multip:
                with Pool() as pool:
                    for itr, stats_events in enumerate(tqdm_redirect(pool.imap(engine, iters_range), total=args.niters)):
                        # Record sim results
                        stats.sim_summary[inet][itr] = stats_events
            else:
                for itr in tqdm_redirect(iters_range):
                    print('Running iteration ' + str(itr) + ':')
                    
                    # Reinitialize network + Random first infected at the beginning of each run BUT the first one
                    # This is needed only in sequential processing since in multiprocessing the nets are deepcopied anyway
                    if itr:
                        engine.reinit_net(first_inf)
                    
                    # Run simulation
                    stats_events = engine(itr)
                    
                    # Record sim results
                    stats.sim_summary[inet][itr] = stats_events
                    print('---> Result:' + str(stats_events[-1]['totalInfected']) + ' total infected persons over time.')
                    
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
        raise NotImplemented
    

class EngineDual(Engine):
    
    def reinit_net(self, first_inf):    
        super().reinit_net(first_inf)
        self.know_net.init_states('S')
        self.know_net.change_state(first_inf, state='I', update=True)
        
    def __call__(self, itr):
        
        # local vars for efficiency
        args = self.args
        true_net = self.true_net
        know_net = self.know_net
        node_list = know_net.node_list

        # Draw network if flag
        if args.draw:
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.title('True Network')
            self.true_net.draw(show=False)
            plt.subplot(122)
            plt.title('Tracing Network')
            self.know_net.draw(self.true_net.pos, show=False)
            plt.show()
            

        # simulation objects
        sim_true = Simulation(true_net, self.trans_true)
        sim_know = Simulation(know_net, self.trans_know)
        
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

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2
                
                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                if e.to == 'T':
                    sim_true.run_trace_event(e, True)
                    sim_know.run_trace_event(e, True)
                # Non-compliance with isolation event
                elif e.fr == 'T':
                    sim_true.run_trace_event(e, False)
                    sim_know.run_trace_event(e, False)
                    m['totalNonCompliant'] += 1
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

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

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
                    
            # event.TO Updates are common for all models
            
            if e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if self.no_exposed: m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRecovered'] += 1
            elif e.to == 'T':
                m['totalTraced'] += 1
                # if S -> T then a person has been incorrectly traced
                if e.fr == 'S': m['totalFalseTraced'] += 1
                if e.fr == 'E': m['totalExposedTraced'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1
                
            # if this changes to False, simulation is continued
            stop_simulation = True
                
            # local vars for efficiency
            node_states = know_net.node_states
            node_traced = know_net.node_traced
            node_counts = know_net.node_counts

            # compute random tracing effort -> we only care about the number of traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            randEffortAcum = 0
            # compute active tracing effort -> we only care about the neighs of nodes in traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            tracingEffortAccum = 0
            for nid in node_list:
                current_state = node_states[nid]
                if not node_traced[nid] and current_state in ['S', 'E', 'I', 'Ia', 'Is']:
                    randEffortAcum += 1
                    tracingEffortAccum += node_counts[nid]['T']
                # if one state is not a final configuration state, continue simulation
                if current_state not in ['S', 'R', 'D']:
                    stop_simulation = False
                    
            m['tracingEffortRandom'] = args.taur * randEffortAcum
            m['tracingEffortContact'] = self.tr_rate * tracingEffortAccum
            
            # False Positives: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalsePositive'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))

            # draw network at each inner state if option selected
#             if args.draw_iter:
#                 print('State after events iteration ' + str(i) + ':')
#                 plt.figure(figsize=(14, 5))
#                 plt.subplot(121)
#                 plt.title('True Network')
#                 self.true_net.draw(show=False)
#                 plt.subplot(122)
#                 plt.title('Tracing Network')
#                 self.know_net.draw(self.true_net.pos, show=False)
#                 plt.show()
#                 sleep(1.5)

            # close simulation if the only possible events remain T -> N -> T
            if stop_simulation:
                break

        if args.draw:
            print('Final state:')
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.title('True Network')
            self.true_net.draw(show=False)
            plt.subplot(122)
            plt.title('Tracing Network')
            self.know_net.draw(self.true_net.pos, show=False)
            plt.show()                

        return result

            
class EngineOne(Engine):
    
    def __call__(self, itr):
        
        if self.args.draw:
            self.true_net.draw()
      
        # simulation objects
        sim_true = Simulation(self.true_net, self.trans_true)
        
        # number of initial infected
        inf = self.args.first_inf

        # metrics to record simulation summary
        m = {
            'nI' : inf,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : inf,
            'totalInfectious': inf,
            'totalTraced' : 0,
            'totalFalseTraced': 0,
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
            
            ### NOTE: Running separate_traced sims on a Single network is actually SLOWER than running on Dual directly
            if self.args.separate_traced:
                # allow for separate tracing events to also be considered in the current events loop
                e2 = sim_true.get_next_trace_event()

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                if e.to == 'T':
                    sim_true.run_trace_event(e)
                # otherwise, normal logic follows (with update counts)
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
                    
            
            # event.TO Updates are common for all models
                                
            if e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if self.no_exposed: m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRecovered'] += 1
            elif e.to == 'T':
                m['totalTraced'] += 1
                # if S -> T then a person has been incorrectly traced
                if e.fr == 'S': m['totalFalseTraced'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1

            # compute random tracing effort -> net_size - non_traceable_states = traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            m['tracingEffortRandom'] = self.args.taur * (self.args.netsize - \
                m['totalTraced'] - m['totalHospital'] - m['totalRecovered'] - m['totalDeath'])
            # compute active tracing effort -> we only care about traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            tracingEffortAccum = 0
            for nid in self.know_net.node_list:
                current_state = self.know_net.node_states[nid]
                if current_state in ['S', 'E', 'I', 'Ia', 'Is']:
                    tracingEffortAccum += self.true_net.node_counts[nid]['T']
            m['tracingEffortContact'] = self.tr_rate * tracingEffortAccum

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))
            
#             if self.args.draw_iter:
#                 print('State after events iteration ' + str(i) + ':')
#                 self.true_net.draw()
            
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