import numpy as np
import matplotlib.pyplot as plt

from time import sleep
from itertools import count
from multiprocessing import cpu_count
from collections import defaultdict
from multiprocess.pool import Pool

from lib import network
from lib.utils import tqdm_redirect, ListDelegator
from lib.stats import StatsEvent

# Classes used to perform Multiprocessing iterations

class Engine():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def reinit_net(self, first_inf_nodes):
        self.true_net.reinit_for_another_iter(first_inf_nodes)
        
    def __call__(self, itr):
        raise NotImplementedError
        
        
class EngineNet(Engine):
    """
    This class will be used for multiprocessing over different network initialization
    """
    
    def __call__(self, inet, return_last_net=False):
        # local vars for efficiency
        args = self.args
        first_inf_nodes = self.first_inf_nodes
        no_exposed = self.no_exposed
        is_covid = self.is_covid
        tr_rate = self.tr_rate
        trans_true_items = self.trans_true_items
        trans_know_items = self.trans_know_items
        
        # will hold all network events
        net_events = defaultdict()
        
        # Get true_net with all nodes in state 'S' but one which is 'I'
        true_net = network.get_random(args.netsize, args.k, args.rem_orphans, args.presample, weighted=False)
        true_net.change_state(first_inf_nodes, state='I', update=True)

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
            jobs = int(cpu_count() / (args.multip - 1))
            with Pool(jobs) as pool:
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
                    engine.reinit_net(first_inf_nodes)

                # Run simulation
                stats_events = engine(itr)

                # Record sim results
                net_events[itr] = stats_events
                print('---> Result:' + str(stats_events[-1]['totalInfected']) + ' total infected persons over time.')
                
        if return_last_net:
            return net_events, true_net, know_net
                
        return net_events

    
class EngineDual(Engine):
    
    def reinit_net(self, first_inf_nodes):    
        super().reinit_net(first_inf_nodes)
        self.know_net.reinit_for_another_iter(first_inf_nodes)
        
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
