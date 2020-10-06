import argparse
import random

from tqdm import tqdm
from time import sleep
from itertools import count
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool as Pool

import matplotlib.pyplot as plt

from lib import network
from lib.utils import tqdm_redirect, expFactorTimesCount
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
    'tautrange': False, # testing multiple values for taut
    'netsize': 1000, 'k': 10, # net size and avg degree
    'overlap': .08, 'zadd': 0, 'zrem': 5, # net overlap translated to z_rem / z_add & z_rem OR specific z_add, z_rem if overlap is None
    'nnets': 1, 'niters': 1, 'nevents': 0, # running number of nets, iterations per net and events (if 0, until no more events)
    'multip': False, # whether multiprocessing is used
    'draw': False, 'draw_iter': False, 'dual': True, 'seed': 43,
    # None -> full_summary never called; False -> no summary printing, True -> print summary as well
    'summary': None,
    'separate_traced': False, # whether to have the Traced state separate from all the other states
    'model': 'sir', # can be sir, seir or covid
    # COVID model specific parameters:
    'pa': 0.2, # probability of being asymptomatic (could also be 0.5)
    'miup': 1/1.5, # duration of prodromal phase
    'ph': [0, 0.1, 0.2], # probability of being hospitalized (i.e. having severe symptoms Pss) based on age category 
    'lamdahr': [0, .083, .033], # If hospitalized, daily rate entering in R based on age category
    'lamdahd': [0, .0031, .0155], # If hospitalized, daily rate entering in D based on age category
    'group': 1 # Age group; Can be 0 - children, 1 - adults, 2 - senior
}

def main(args):
    
    # Will hold stats for all simulations
    stats = StatsProcessor(args)
    
    # seed the random
    random.seed(args.seed)
    
    # Random first infected across simulations
    first_inf = random.sample(range(args.netsize), 1)[0]
    
    # Boolean responsible for determining whether nInfectious = nInfected
    no_exposed = (args.model == 'sir')
    
    # Whether the model is Covid or not
    is_covid = (args.model == 'covid')
 
    # Set recovery rate for traced people based on whether gammatau was provided
    if args.gammatau is None: args.gammatau = args.gamma
    
    # Transition dictionaries for each network will be populated based on args.model {state->list(state, trans_func)}
    trans_true_items, trans_know_items = get_transitions_for_model(args)
    
    # we can simulate with a range of tracing rates or with a single one provied by args.taut
    tracing_rates = [0, .1, .25, .5, .75, 1, 1.5, 2.5] if args.tautrange else [args.taut]
    
    for tr_rate in tracing_rates:
        
        print('For taut =', tr_rate)

        # Tracing for 'S', 'E', 'I(p)' happens over know_net depending only on the traced neighbor count of nid
        tr_func = lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=0)
        add_trans(trans_know_items, 'S', 'T', tr_func)
        add_trans(trans_know_items, 'E', 'T', tr_func)
        add_trans(trans_know_items, 'I', 'T', tr_func)
        
        # Tracing for 'Ia' and 'Is' also depends on a random tracing rate (due to random testing)
        tr_and_test_func = lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=args.taur)
        add_trans(trans_know_items, 'Ia', 'T', tr_and_test_func)
        add_trans(trans_know_items, 'Is', 'T', tr_and_test_func)
                
        for inet in range(args.nnets):
            print('Simulating network - No.', inet)
            
            # Get true_net with all nodes in state 'S' but one which is 'I'
            true_net = network.get_random(args.netsize, args.k)
            true_net.change_state(first_inf, state='I', update=True)
                        
            if args.dual:
                # Priority will be given to the overlap value
                # Normally z_add = z_rem = z, but if z_add = 0 then only z_rem=z is produced
                # If overlap is None, zadd and zrem values are used
                know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem)
                
                # Object used during Multiprocessing of Network simulation events
                engine = EngineDual(
                    args=args, no_exposed=no_exposed, is_covid=is_covid,
                    true_net=true_net, know_net=know_net,
                    trans_true=trans_true_items, trans_know=trans_know_items
                )
                
            else:
                # Object used during Multiprocessing of Network simulation events
                engine = EngineOne(
                    args=args, no_exposed=no_exposed, is_covid=is_covid,
                    true_net=true_net,
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
        
    if args.summary is not None:
        return stats.full_summary(args.summary)

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

        # Draw network if flag
        if self.args.draw:
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.title('True Network')
            self.true_net.draw(show=False)
            plt.subplot(122)
            plt.title('Tracing Network')
            self.know_net.draw(self.true_net.pos, show=False)
            plt.show()        

        # simulation objects
        sim_true = Simulation(self.true_net, self.trans_true)
        sim_know = Simulation(self.know_net, self.trans_know)

        # metrics to record simulation summary
        m = {
            'nI' : 1,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : 1,
            'totalInfectious': 1,
            'totalTraced' : 0,
            'totalRecovered' : 0,
            'totalHospital' : 0,
            'totalDeath' : 0,
            'tracingEffortRandom' : -1,
            'tracingEffortContact' : -1,
        }
        
        # results over time recorded
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run exactly args.nevents otherwise
        events_range = range(self.args.nevents) if self.args.nevents else count()
        for i in events_range:
            
            e1 = sim_true.get_next_event()
            
            if self.args.separate_traced:
                e2 = sim_know.get_next_trace_event()

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

                sim_true.run_event_separate_traced(e)
                sim_know.run_event_separate_traced(e)
                
                if e.to != 'T':
                    # update simulation statistics via event.FROM only if event was NOT a tracing event
                    # for models other than covid, leaving 'I' means a decrease in infectious
                    # for covid, leaving Ia or Is state means current infectious decreases
                    if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                        m['nI'] -= 1
                    elif e.fr == 'E':
                        m['nE'] -= 1
                    elif e.fr == 'H':
                        m['nH'] -= 1
                    
            else:
                e2 = sim_know.get_next_event()

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

                sim_true.run_event(e)
                sim_know.run_event(e)
                
                # if NOT separate_traced, then update statistics is as-normal for all events (incl tracing)
                # I -> T OR leaving Ia/Is means a decrease in number of infecious at the time
                if e.fr == 'I' and e.to == 'T' or e.fr == 'Ia' or e.fr == 'Is':
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
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1

            # compute random tracing effort -> net_size - non_traceable_states = traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            m['tracingEffortRandom'] = self.args.taur * 
                (self.args.netsize - m['totalTraced'] - m['totalHospital'] - m['totalRecovered'] - m['totalDeath'])
            # compute active tracing effort -> we only care about traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            tracingEffortAccum = 0
            for nid, state in enumerate(self.know_net.node_states):
                if state in ['S', 'E', 'I', 'Ia', 'Is']:
                    tracingEffortAccum += self.know_net.node_counts[nid]['T']
            m['tracingEffortContact'] = self.args.taut * tracingEffortAccum

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))

            # draw network at each inner state if option selected
#             if self.args.draw_iter:
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

        if self.args.draw:
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

        # metrics to record simulation summary
        m = {
            'nI' : 1,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : 1,
            'totalInfectious': 1,
            'totalTraced' : 0,
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

            e = sim_true.get_next_event()

            # If no more events left, break out of the loop
            if e is None:
                break

            sim_true.run_event(e)

            # I -> T OR leaving Ia/Is means a decrease in number of infecious at the time
            if e.fr == 'I' and e.to == 'T' or e.fr == 'Ia' or e.fr == 'Is':
                m['nI'] -= 1
            elif e.fr == 'E':
                m['nE'] -= 1
            elif e.fr == 'H':
                m['nH'] -= 1
                                
            if e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if self.no_exposed: m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRecovered'] += 1
            elif e.to == 'T':
                m['totalTraced'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'H':
                m['nH'] += 1
                m['totalHospital'] += 1
            elif e.to == 'D':
                m['totalDeath'] += 1

            # compute random tracing effort -> net_size - non_traceable_states = traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            m['tracingEffortRandom'] = self.args.taur * (self.args.netsize 
                                                         - m['totalTraced'] - m['totalHospital'] - m['totalRecovered'] - m['totalDeath'])
            # compute active tracing effort -> we only care about traceable_states ('S', 'E', 'I', 'Ia', 'Is')
            tracingEffortAccum = 0
            for nid, state in enumerate(self.know_net.node_states):
                if state in ['S', 'E', 'I', 'Ia', 'Is']:
                    tracingEffortAccum += self.know_net.node_counts[nid]['T']
            m['tracingEffortContact'] = self.args.taut * tracingEffortAccum

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))
            
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
        argparser.add_argument('--' + k, type=type(default), default=default)

    args = argparser.parse_args()
    args.summary = True # If script run, full_summary in print mode will always be called

    main(args)