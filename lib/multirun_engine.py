import random
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
        self.true_net.init_for_simulation(first_inf_nodes)
        
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
        
        # initialize seeds to None
        net_seed = dual_seed = tri_seed = None
        # the net random seed will be corresponding to its index (if any netseed selected at all)
        if args.netseed is not None:
            net_seed = args.netseed + inet
            dual_seed = net_seed + 1
            tri_seed = net_seed + 2
        
        # Get true_net with all nodes in state 'S' but one which is 'I'
        true_net = network.get_random(args.netsize, args.k, args.rem_orphans, typ=args.nettype,
                                      p=args.p, weighted=False, seed=net_seed, inet=inet)
        true_net.change_state(first_inf_nodes, state='I', update=True)

        # Placeholder for the dual network (will be initialized only if args.dual)
        know_net = None

        if args.dual:
            # First dual net depends on both overlap and uptake (this is usually the digital contact tracing net)
            know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem, seed=dual_seed,
                            keep_nodes_percent=args.uptake, maintain_overlap=args.maintain_overlap, count_importance=tr_rate)

            # if 2 dual networks selected, create the second network and add both to a ListDelegator
            if args.dual == 2:
                # Second net depends only on overlap_two - i.e. uptake = 1 (this is usually the manual tracing net)
                know_net_two = network.get_dual(true_net, args.overlap_two, args.zadd_two, args.zrem_two, seed=tri_seed,
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
        self.know_net.init_for_simulation(first_inf_nodes)
        
    def __call__(self, itr):
        
        # local vars for efficiency
        args = self.args
        dual = args.dual
        tr_rate = self.tr_rate
        presample = args.presample
        separate_traced = args.separate_traced
        samp_already_exp = (args.sampling_type == 'dir')
        samp_min_only = (args.sampling_type == 'min')
        noncomp_after = args.noncomp_after
        taur = args.taur
        true_net = self.true_net
        # Note: know_net may be a ListDelegator of tracing networks
        know_net = self.know_net
        # we need the tracing networks in this form to pass to get_next_event_sample_only_minimum
        tracing_nets = [know_net] if dual == 1 else list(know_net)
        # the transition objects
        trans_true = self.trans_true
        trans_know = self.trans_know
        
        # seed the random for this network id and iteration
        if args.seed is not None:
            seeder = args.seed + true_net.inet + itr
            random.seed(seeder)
            np.random.seed(seeder)
            
        # drawing vars
        draw = args.draw
        draw_iter = args.draw_iter
        animate = args.animate
        
        # If either of the drawing flag is enabled, instantiate drawing figure and axes, and draw initial state
        if draw or draw_iter:
            # import IPython here to avoid dependency if no drawing is performed
            from IPython import display

            fig, ax = plt.subplots(nrows=1, ncols=int(dual) + 1, figsize=(14, 5))

            # we make plots for the true network + all the dual networks
            ax[0].set_title('True Network', fontsize=14)
            true_net.draw(layout_type=args.draw_layout, seed=args.netseed, show=False, ax=ax[0])
            ax[1].set_title('Digital Tracing Network', fontsize=14)
            if dual == 2:
                ax[-1].set_title('Manual Tracing Network', fontsize=14)
            know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], full_name=args.draw_fullname, model=args.model)
            display.display(plt.gcf())
            if animate:
                display.clear_output(wait=True)

        # simulation objects
        sim_true = true_net.get_simulator(trans_true, isolate_S=args.isolate_s, trace_once=args.trace_once, \
                                          already_exp=samp_already_exp, presample=presample)
        sim_know = know_net.get_simulator(trans_know, isolate_S=args.isolate_s, trace_once=args.trace_once, \
                                          already_exp=samp_already_exp, presample=presample)
        
        # number of initial infected
        inf = args.first_inf

        # metrics to record simulation summary
        m = {
            'nI' : inf,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : inf,
            'totalInfectious': inf,
            'totalFalseNegative': 0,
            'totalTraced' : 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered' : 0,
            'totalHospital' : 0,
            'totalDeath' : 0,
            'tracingEffortRandom' : 0,
            'tracingEffortContact' : [0],
        }
        
        # results over time recorded
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run exactly args.nevents otherwise
        events_range = range(args.nevents) if args.nevents else count()
        for i in events_range:
            
            # this option enables T/N states to be separate from the infection states
            if separate_traced:
                
                # if samp_min_only then we only sample the minimum exponential Exp(sum(lamdas[i] for i in possible_transitions))
                if samp_min_only:
                    # we combine all possible lambdas from both infection and tracing networks (so pass tracing-related objs here)
                    e = sim_true.get_next_event_sample_only_minimum(trans_know, tracing_nets)

                # otherwise, all exponentials are sampled as normal (either they are already exp or will be sampled later)
                else:
                    e1 = sim_true.get_next_event()
                    # get events on the known network ONLY IF a random tracing rate actually exists
                    e2 = sim_know.get_next_trace_event() if taur else None

                    # If no more events left, break out of the loop
                    if e1 is None and e2 is None:
                        break

                    # get all event candidates, assign 'inf' to all Nones, and select the event with the smallest 'time' value
                    # Note: if 2 networks are used, e2 will be an array of 2 tracing events from each network
                    candidates = np.append(np.atleast_1d(e2), e1)
                    times = [e_candidate.time if e_candidate is not None else float('inf') for e_candidate in candidates]
                    e = candidates[np.argmin(times)]

                                                
                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                is_trace_event = (e.to == 'T')
                # exception: if trace_h selected, the node will be counted as traced if going to H (but will never become noncompliant)
                if is_trace_event or (args.trace_h and e.to == 'H'):
                    # the update to total traced counts is done only once (ignore if same nid is traced again)
                    if e.node not in true_net.traced_time:
                        m['totalTraced'] += 1
                        # if S -> T then a person has been incorrectly traced
                        if e.fr == 'S': m['totalFalseTraced'] += 1
                        if e.fr == 'E': m['totalExposedTraced'] += 1
                    
                    sim_true.run_trace_event(e, True)
                    sim_know.run_trace_event(e, True)
                    
                # Non-compliance with isolation event
                if e.to == 'N':
                    # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                    if e.node not in true_net.noncomp_time:
                        m['totalNonCompliant'] += 1
                    
                    sim_true.run_trace_event(e, False)
                    sim_know.run_trace_event(e, False)
                    
                # otherwise, normal logic follows if NOT e.to=T event (with update counts on infection network only)
                elif not is_trace_event:
                    sim_true.run_event(e)
                    sim_know.run_event_no_update(e)
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
                e1 = sim_true.get_next_event()
                # get events on the known network ONLY IF a random tracing rate actually exists
                e2 = sim_know.get_next_event() if taur else None

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
                if e.to == 'T' or (args.trace_h and e.to == 'H'):
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
            
            # False Neg: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalseNegative'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])
            
            
            current_time = e.time
            # if args.noncomp_after selected, nodes that have been isolating for longer than noncomp_after become automatically N
            if noncomp_after:
                node_traced = true_net.node_traced
                node_states = true_net.node_states
                traced_time = true_net.traced_time
                blocked_states = ['H', 'D']
                for nid in true_net.node_list:
                    if node_traced[nid] and node_states[nid] not in blocked_states \
                    and (current_time - traced_time[nid] >= noncomp_after):
                        event = sim_true.get_event_with_config(node=nid, fr='T', to='N', time=current_time)
                        sim_true.run_trace_event(event, False)
                        sim_know.run_trace_event(event, False)                          
            
            # if args.efforts selected, compute efforts for all tracing networks
            if args.efforts:
                # loop through nodes to collect list of tracing efforts FOR EACH tracing network
                efforts_for_all = np.atleast_2d(know_net.compute_efforts(taur))
                # random (testing) effort will be the same for all dual networks - element 0 in each result list      
                m['tracingEffortRandom'] = efforts_for_all[0, 0]
                # the contact tracing effort is DIFFERENT across the dual networks - element 1 in each result list
                m['tracingEffortContact'] = efforts_for_all[:, 1]
            
            # draw network at the end of each inner state if option selected
            if draw_iter:
                print('State after events iteration ' + str(i) + ':')
                clear_axis(ax)
                ax[0].set_title('True Network', fontsize=14)
                true_net.draw(layout_type=args.draw_layout, seed=args.netseed, show=False, ax=ax[0])
                ax[1].set_title('Digital Tracing Network', fontsize=14)
                if dual == 2:
                    ax[-1].set_title('Manual Tracing Network', fontsize=14)
                know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], full_name=args.draw_fullname, model=args.model)
                
                display.display(plt.gcf())
                if animate:
                    display.clear_output(wait=True)
                    
                sleep(int(args.draw_iter))
                
            # record metrics after event run for time current_time=e.time
            m['time'] = current_time
            result.append(StatsEvent(**m))
            # close simulation if there are no more possible events on the infection network
            if not m['nE'] + m['nI'] + m['nH']:
                break
                
        if draw:
            # Drawing positions are already initialized by this point
            print('Final state:')
            clear_axis(ax)
            ax[0].set_title('True Network', fontsize=14)
            true_net.draw(show=False, ax=ax[0])
            ax[1].set_title('Digital Tracing Network', fontsize=14)
            if dual == 2:
                ax[-1].set_title('Manual Tracing Network', fontsize=14)
            know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], full_name=args.draw_fullname, model=args.model)
            
            display.display(plt.gcf())
            if animate:
                display.clear_output(wait=True)
                
            if draw == 2:
                plt.savefig('fig/network-' + str(true_net.inet) + '.pdf', format='pdf', bbox_inches = 'tight')
            
        if not animate:
            plt.close()
        return result

            
class EngineOne(Engine):
    
    def __call__(self, itr):
        
        # local vars for efficiency
        args = self.args
        noncomp_after = args.noncomp_after
        taur = args.taur
        separate_traced = args.separate_traced
        true_net = self.true_net
        trans = self.trans_true
        
        # drawing vars
        draw = args.draw
        draw_iter = args.draw_iter
        animate = args.animate
        
        if draw or draw_iter:
            # import IPython here to avoid dependency if no drawing is performed
            from IPython import display
            
            fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
            ax[0].set_title('Infection Progress', fontsize=14)
            true_net.is_dual = False
            true_net.draw(layout_type=args.draw_layout, seed=args.netseed, show=False, ax=ax[0])
            ax[1].set_title('Tracing Progress', fontsize=14)
            true_net.is_dual = True
            true_net.draw(show=False, ax=ax[1], full_name=args.draw_fullname, model=args.model)
            display.display(plt.gcf())
            if animate:
                display.clear_output(wait=True)
            
        # set the dual flag such that true_net can be interpreted as the tracing network
        true_net.is_dual = True
            
        # seed the random for this network id and iteration
        if args.seed is not None:
            seeder = args.seed + true_net.inet + itr
            random.seed(seeder)
            np.random.seed(seeder)
        
        # in the case of separate_traced, mimic dual simulation objects as if dual net was run (sim_true + sim_know)
        if separate_traced:
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
        inf = args.first_inf

        # metrics to record simulation summary
        m = {
            'nI' : inf,
            'nE' : 0,
            'nH' : 0,
            'totalInfected' : inf,
            'totalInfectious': inf,
            'totalFalseNegative': 0,
            'totalTraced' : 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered' : 0,
            'totalHospital' : 0,
            'totalDeath' : 0,
            'tracingEffortRandom' : 0,
            'tracingEffortContact' : [0],
        }

        # will hold up all results across time
        result = []

        # Infinite loop if args.nevents not specified (run until no event possible) OR run args.nevents otherwise
        events_range = range(args.nevents) if args.nevents else count()

        for i in events_range:

            e1 = sim_true.get_next_event()
            
            ### NOTE: Running sims with this option on a Single network is actually SLOWER than running on Dual net directly
            if separate_traced:
                # allow for separate tracing events to also be considered in the current events loop
                e2 = sim_know.get_next_trace_event()

                # If no more events left, break out of the loop
                if e1 is None and e2 is None:
                    break

                e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2
                
                # update the time for the 'fake' sim_know object
                sim_know.time = e.time

                # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                is_trace_event = (e.to == 'T')
                # exception: if trace_h selected, the node will be counted as traced if going to H (but will never become noncompliant)
                if is_trace_event or (args.trace_h and e.to == 'H'):
                    # the update to total traced counts is done only once (ignore if same nid is traced again)
                    if e.node not in true_net.traced_time:
                        m['totalTraced'] += 1
                        # if S -> T then a person has been incorrectly traced
                        if e.fr == 'S': m['totalFalseTraced'] += 1
                        if e.fr == 'E': m['totalExposedTraced'] += 1
                    
                    sim_true.run_trace_event(e, True)
                    
                # Non-compliance with isolation event
                if e.to == 'N':
                    # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                    if e.node not in true_net.noncomp_time:
                        m['totalNonCompliant'] += 1
                        
                    sim_true.run_trace_event(e, False)
                
                # otherwise, normal logic follows if NOT e.to=T event (with update counts on infection network only)
                elif not is_trace_event:
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
                    
                # Update 'T' e.fr counts
                if e.to == 'T' or (args.trace_h and e.to == 'H'):
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

            # False Neg: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalseNegative'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])
            
            
            # if args.noncomp_after selected, nodes that have been isolating for longer than noncomp_after become automatically N
            current_time = e.time
            
            if noncomp_after and separate_traced:
                node_traced = true_net.node_traced
                node_states = true_net.node_states
                traced_time = true_net.traced_time
                blocked_states = ['H', 'D']
                for nid in true_net.node_list:
                    if node_traced[nid] and node_states[nid] not in blocked_states \
                    and (current_time - traced_time[nid] >= noncomp_after):
                        event = sim_true.get_event_with_config(node=nid, fr='T', to='N', time=current_time)
                        sim_true.run_trace_event(event, False)
                        sim_know.run_trace_event(event, False)
            
            # if args.efforts selected, compute efforts for all tracing networks
            if args.efforts:
                # loop through nodes to collect list of tracing efforts FOR EACH tracing network
                efforts_for_all = true_net.compute_efforts(taur)
                # random (testing) effort is the first element (no dual networks here)    
                m['tracingEffortRandom'] = efforts_for_all[0]
                # the contact tracing effort is the first second (no dual networks here) 
                m['tracingEffortContact'] = [efforts_for_all[1]]
                
            
            if draw_iter:
                print('State after events iteration ' + str(i) + ':')
                clear_axis(ax)
                ax[0].set_title('Infection Progress', fontsize=14)
                true_net.is_dual = False
                true_net.draw(layout_type=args.draw_layout, seed=args.netseed, show=False, ax=ax[0])
                ax[1].set_title('Tracing Progress', fontsize=14)
                true_net.is_dual = True
                true_net.draw(show=False, ax=ax[1], full_name=args.draw_fullname, model=args.model)
                display.display(plt.gcf())
                if animate:
                    display.clear_output(wait=True)
                    
                sleep(int(args.draw_iter))

                
            # record metrics after event run for time current_time=e.time
            m['time'] = current_time
            result.append(StatsEvent(**m))
            # close simulation if there are no more possible events on the infection network
            if not m['nE'] + m['nI'] + m['nH']:
                break
            
        if draw:
            print('Final state:')
            clear_axis(ax)
            ax[0].set_title('Infection Progress', fontsize=14)
            true_net.is_dual = False
            true_net.draw(show=False, ax=ax[0])
            ax[1].set_title('Tracing Progress', fontsize=14)
            true_net.is_dual = True
            true_net.draw(show=False, ax=ax[1], full_name=args.draw_fullname, model=args.model)
            
            display.display(plt.gcf())
            if animate:
                display.clear_output(wait=True)
                
            if draw == 2:
                plt.savefig('fig/network-' + str(true_net.inet) + '.pdf', format='pdf', bbox_inches = 'tight')
            
        return result


def clear_axis(ax):
    for axis in ax:
        axis.clear()