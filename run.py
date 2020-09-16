import argparse
import random
import tqdm
import math

from time import sleep
from itertools import count
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy

import matplotlib.pyplot as plt

from lib import network
from lib.utils import expFactorTimesCount, tqdm_redirect
from lib.simulation import Simulation
from lib.stats import StatsProcessor, StatsEvent

PARAMETER_DEFAULTS = {
    'beta': 0.1, 'alpha': 1/3.7, 'gamma': 1/(1.5 + 2.3), 
    'taur': 0.1, 'taut': 0.1, 'tautrange': False, 'gammatau': 0.5, 
    'netsize': 1000, 'k': 10, 'draw': False,
    'dual': True, 'overlap': .08, 'zadd': 5, 'zrem': 5,
    'nnets': 1, 'niters': 1, 'nevents': 20, 'seed': 43, 
    'multip': False, 'exposed': False, 'spontan': False,
}

def main(args):
    
    # Will hold stats for all simulations
    stats = StatsProcessor(args)
    
    # seed the random
    random.seed(args.seed)
    
    # Random first infected across simulations
    first_inf = random.sample(range(args.netsize), 1)[0]
     
    # Transition parameters for true_net (only S->E->I->R in Dual net scenario)
    trans_true = defaultdict(dict)
    
    if args.exposed:
        # SEIR model
        # Infections spread based on true_net connections depending on nid
        trans_true['S']['E'] = \
            lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta, base=0)
        # Next transition is network independent (at rate alpha) but we keep the same API for sampling at get_next_event time
        trans_true['E']['I'] = \
            lambda net, nid: -(math.log(random.random()) / args.alpha)
    else:
        # SIR model
        trans_true['S']['I'] = \
            lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta, base=0)
        
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        trans_true['I']['R'] = \
            lambda net, nid: -(math.log(random.random()) / args.gamma)
      
    # Transition parameters for know_net (only I->T->R in Dual net scenario)
    # If args.dual false, the same net and transition objects are used for both infection and tracing
    trans_know = defaultdict(dict) if args.dual else trans_true
    # Recovery for traced nodes is network independent at rate gammatau
    trans_know['T']['R'] = \
        lambda net, nid: -(math.log(random.random()) / args.gammatau)
    
    # we cache transition items due to efficiency reasons
    trans_true_items = defaultdict(list, {k : list(trans_true[k].items()) for k in trans_true})
    trans_know_items = defaultdict(list, {k : list(trans_know[k].items()) for k in trans_know}) if args.dual else trans_true_items
    
    # we can simulate with a range of tracing rates or with a single one provied by args.taut
    tracing_rates = [0, .1, .25, .5, .75, 1, 1.5, 2.5] if args.tautrange else [args.taut]
    
    for tr_rate in tracing_rates:
        
        print('For taut =', tr_rate)
    
        # Tracing happens over know_net depending on nid
        trans_know['I']['T'] = \
            lambda net, nid: expFactorTimesCount(net, nid, state='T', lamda=tr_rate, base=args.taur)
        # we cache trans_know_items due to efficiency reasons
        trans_know_items['I'] = list(trans_know['I'].items())
                
        for inet in range(args.nnets):
            print('Simulating network - No.', inet)
            
            # Get true_net with all nodes in state 'S' but one which is 'I'
            true_net = network.get_random(args.netsize, args.k)
            true_net.change_state(first_inf, state='I', update=True) 
            
            if args.dual:
                # Priority will be given to the overlap value -> z_add = z_rem = z
                # If overlap is None, zadd and zrem values are used
                know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem)

                # Object used during Multiprocessing of Network simulation events
                engine = EngineDual(
                    args=args,
                    true_net=true_net, know_net=know_net,
                    trans_true=trans_true_items, trans_know=trans_know_items
                )
                
            else:
                # Object used during Multiprocessing of Network simulation events
                engine = EngineOne(
                    args=args,
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
                    
                    # Reinitialize network + Random first infected at the beginning of each run but the first one
                    # This is needed only in sequential processing since in multiprocessing the nets are deepcopied anyway
                    if itr:
                        engine.reinit_net(first_inf)
                    
                    # Run simulation
                    stats_events = engine(itr)
                    
                    # Record sim results
                    stats.sim_summary[inet][itr] = stats_events
                    print('---> Result:' + str(stats_events[-1]['totalInfected']) + ' total infected persons over time.')
                    
        stats.results_for_param(tr_rate)

    return stats     
        

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
            self.true_net.draw(show=False)
            plt.subplot(122)
            self.know_net.draw(self.true_net.pos, show=False)
            plt.show()        

        # simulation objects
        sim_true = Simulation(self.true_net, self.trans_true)
        sim_know = Simulation(self.know_net, self.trans_know)

        # metrics to record simulation summary
        m = {
            'nE' : 0,
            'nI' : 1,
            'totalInfected' : 1,
            'totalInfectious': 1,
            'totalTraced' : 0,
            'totalRemoved' : 0,
            'tracingEffortRandom' : -1,
            'tracingEffortContact' : -1,
        }
        
        # results over time recorded
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run exactly args.nevents otherwise
        events_range = range(self.args.nevents) if self.args.nevents else count()
        for i in events_range:
            e1 = sim_true.get_next_event()
            e2 = sim_know.get_next_event()

            # If no more events left, break out of the loop
            if e1 is None and e2 is None:
                break

            e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

            sim_true.run_event(e)
            sim_know.run_event(e)

            # update counts
            if e.fr == 'I':
                m['nI'] -= 1
            elif e.fr == 'E':
                m['nE'] -= 1
            if e.to == 'T':
                m['totalTraced'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if not self.args.exposed:
                    m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRemoved'] += 1

            # compute random tracing effort -> netsize - totalTraced - nTotalRemoved = 'S' + 'E' + 'I'
            m['tracingEffortRandom'] = self.args.taur * (self.args.netsize - m['totalTraced'] - m['totalRemoved'])
            # compute active tracing effort -> still care only about 'S', 'E', 'I'
            tracingEffortAccum = 0
            for nid, state in enumerate(self.know_net.node_states):
                if state in ['S', 'E', 'I']:
                    tracingEffortAccum += self.know_net.node_counts[nid]['T']
            m['tracingEffortContact'] = self.args.taut * tracingEffortAccum

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))

            # draw network at each inner state if option selected
#             if self.args.draw:
#                 print('State after events iteration', i, ':')
#                 plt.figure(figsize=(14, 5))
#                 plt.subplot(121)
#                 self.true_net.draw(show=False)
#                 plt.subplot(122)
#                 self.know_net.draw(self.true_net.pos, show=False)
#                 plt.show()
#                 sleep(1.5)

        if self.args.draw:
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.true_net.draw(show=False)
            plt.subplot(122)
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
            'nE' : 0,
            'nI' : 1,
            'totalInfected' : 1,
            'totalInfectious': 1,
            'totalTraced' : 0,
            'totalRemoved' : 0,
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

            # update counts
            if e.fr == 'I':
                m['nI'] -= 1
            elif e.fr == 'E':
                m['nE'] -= 1
            if e.to == 'T':
                m['totalTraced'] += 1
            elif e.to == 'E':
                m['nE'] += 1
                m['totalInfected'] += 1
            elif e.to == 'I':
                m['nI'] += 1
                m['totalInfectious'] +=1
                # in SIR totalInfected = totalInfectious
                if not self.args.exposed:
                    m['totalInfected'] = m['totalInfectious']
            elif e.to == 'R':
                m['totalRemoved'] += 1

            # compute random tracing effort -> netsize - totalTraced - nTotalRemoved = 'S' + 'E' + 'I'
            m['tracingEffortRandom'] = self.args.taur * (self.args.netsize - m['totalTraced'] - m['totalRemoved'])
            # compute active tracing effort -> still care only about 'S', 'E', 'I'
            tracingEffortAccum = 0
            for nid, state in enumerate(self.true_net.node_states):
                if state in ['S', 'E', 'I']:
                    tracingEffortAccum += self.true_net.node_counts[nid]['T']
            m['tracingEffortContact'] = self.args.taut * tracingEffortAccum

            # record metrics after event run for time e.time
            m['time'] = e.time
            result.append(StatsEvent(**m))
            
        if self.args.draw:
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

    main(args)