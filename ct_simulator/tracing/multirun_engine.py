import random
import numpy as np
import dill
import matplotlib.pyplot as plt

from math import ceil
from time import sleep
from itertools import count
from collections import defaultdict, Counter

from . import network
from .stats import StatsEvent
from .models import add_trans
from .utils import get_pool, tqdm_redirect, is_not_empty, no_std_context, get_stateless_sampling_func, get_stateful_sampling_func, ListDelegator


NETWORK_TITLE = 'True Network'
NETWORK_TITLE_DUAL = 'Contact Tracing Network'
NETWORK_TITLE_DUAL_TWO = 'Digital Tracing Network'
NETWORK_TITLE_DUAL_THREE = 'Manual Tracing Network'


class Engine():
    """
    The Engine abstract class defines the interface for callable objects that can be used to run multiprocessing simulations.
    The arguments passed to `__init__` become part of the callable object's context, being accessible via dot notation.
    The `__call__` method can then be called with an network/iteration number to run the simulation, with the context being accessible via `self`.
    
    Notes: 
        `__getstate__` and `__setstate__` can be overridden to modify the default pickle logic to avoid standard pickling errors (e.g. lambda functions).
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        # use create byte representation using dill to avoid lambda errors of pickle
        for key in ('trans_true', 'trans_know'):
            if key in state:
                state[key] = dill.dumps(state[key])
        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, state):
        # re-create byte representations using dill to avoid lambda errors of pickle
        for key in ('trans_true', 'trans_know'):
            if key in state:
                state[key] = dill.loads(state[key])
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(state)

    def __call__(self, itr, **kwargs):
        raise NotImplementedError
        
        
class EngineNet(Engine):
    """
    This class extends the Engine abstract class, defining the context-imbued callable to be used for running simulations in parallel 
    over different network initialization.
    """
    def __call__(self, inet, return_last_net=False, **kwargs):
        """
        Run the simulation for a given network ID, used as a seed for the network generation process.

        Args:
            inet (int): The network ID used as a generation seed.
            return_last_net (bool, optional): Whether to return the last network used in the simulation, generally used for debugging.
                This should be True only within serial processing. Defaults to False.
            **kwargs (dict, optional): Additional keyword arguments to pass to the simulation.
        Returns:
            results (dict): A dictionary containing the results of the simulation.
        """
        # local vars for efficiency
        args = self.args
        nettype = args.nettype
        first_inf_nodes = self.first_inf_nodes
        no_exposed = self.no_exposed
        is_covid = self.is_covid
        tr_rate = self.tr_rate
        trans_true = self.trans_true
        trans_know = self.trans_know
        
        # will hold all network events
        net_events = defaultdict()
        # initialize the true network seed either randomly, or based on what has been supplied already + net index
        net_seed = random.randint(0, 1e9) if args.netseed is None else args.netseed + inet
            
        args_dict = vars(args)
             
        # the infection net is either predefined in the first element of args.nettype or created at random based on the model name specified by the same parameter
        # if the net is predefined, we also seed here the first_inf_nodes (this is because we did not have access to the list of nodes prior to this point)
        try:
            true_net = network.get_from_predef(nettype['0'][0], nseed=net_seed, inet=inet, W_factor=nettype.get('Wi', 0), ids=nettype.get('nid', None),
                                               count_importance=tr_rate, **args_dict)
        # TypeError: nettype is a str, KeyError: key '0' is not in the dynamic dict, IndexError: element 0 (inf net) not in list
        except (TypeError, KeyError, IndexError):
            # args_dict should also contain the netsize and the average degree
            true_net = network.get_random(typ=nettype, nseed=net_seed, inet=inet, count_importance=tr_rate, **args_dict)

        # calculate true average degree
        args.avg_deg = true_net.avg_degree()
        # if we didn't set the `netsize` by this point, it means 'nid' was not supplied in the `nettype` dict
        # hence, we assume only the nodes in the edges supplied are part of the network, setting `netsize` accordingly
        if args.netsize <= 0:
            args.netsize = len(true_net.nodes)
        # if `k` is still not set to an eligible value, we can safely set it to `args.avg_deg`
        if args.k <= 0:
            args.k = args.avg_deg
        # turn first_inf into an absolute number if it's a percentage by this point (happens only if predefined net with no nid)
        args.first_inf = int(args.first_inf if args.first_inf >= 1 else args.first_inf * args.netsize)
        if first_inf_nodes != []:
            # first_inf_nodes could have been calculated by this point if an infseed was supplied, and
            # we deal with random network OR 'nid' key was supplied in the predefined network of args.nettype
            if first_inf_nodes is None:
                # Random first infected across simulations - seed random locally
                first_inf_nodes = random.Random(net_seed).sample(true_net.graph.get('partition', true_net.nodes), args.first_inf)
            # Change the state of the first_inf_nodes to 'I' to root the simulation
            true_net.change_state(first_inf_nodes, state='I', update=True)
            
        # Placeholder for the dual network (will be initialized iff args.dual > 0)
        know_net = None

        if args.dual:
            # the dual network is either predefined in the second element of args.nettype or initialized at random
            try:
                know_net = network.get_dual_from_predef(true_net, nettype['0'][1], count_importance=tr_rate, 
                                                        w_factor=nettype.get('Wt', 0), **args_dict)
            except (TypeError, KeyError, IndexError):
                # First dual net depends on both overlap and uptake as this is usually the digital contact tracing net.
                # Note this will also copy over the states, so no need to call change_state
                know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem, args.uptake, args.maintain_overlap,
                                            nseed=net_seed+1, inet=inet, count_importance=tr_rate, **args_dict)
            # update the overlap on the arguments with the actual overlap used by `know_net`
            args.overlap = know_net.overlap

            # if 2 dual networks selected, create the second network and add both to a ListDelegator
            if args.dual == 2:
                try:
                    know_net_two = network.get_dual_from_predef(true_net, nettype['0'][2], count_importance=args.taut_two,
                                                                w_factor=nettype.get('Wt2', 0), **args_dict)
                except (TypeError, KeyError, IndexError):
                    # Second tracing net may attempt to maintain overlap_two as this is usually the manual tracing net. 
                    # Note this will also copy over the states, so no need to call change_state
                    know_net_two = network.get_dual(true_net, args.overlap_two, args.zadd_two, args.zrem_two, 
                                                    args.uptake_two, args.maintain_overlap_two,
                                                    nseed=net_seed+2, inet=inet, count_importance=args.taut_two, **args_dict)
                # update the overlap on the arguments with the actual overlap used by `know_net_two`
                args.overlap_two = know_net_two.overlap
                # know_net becomes a ListDelegator of the 2 networks
                know_net = ListDelegator(know_net, know_net_two)

            # Object used during Multiprocessing of Network simulation events
            engine = EngineDual(
                args=args, no_exposed=no_exposed, is_covid=is_covid,
                true_net=true_net, know_net=know_net,
                trans_true=trans_true, trans_know=trans_know
            )

        else:
            # Object used during Multiprocessing of Network simulation events
            engine = EngineOne(
                args=args, no_exposed=no_exposed, is_covid=is_covid,
                true_net=true_net,
                trans_true=trans_true
            )

        niters = args.niters
        iters_range = range(niters)

        if args.multip == 2 or args.multip == 3:
            # allocate EITHER half or all cpus to multiprocessing for parallelizing simulations for different iterations of 1 init
            # multip == 2 parallelize only iterations; multip == 3 parallelize both net and iters
            jobs = args.cpus // (args.multip - 1)
            pool_type = get_pool(pytorch=args.is_learning_agent, daemon=True, set_spawn=args.control_gpu)
            # for agents, reduce memory burden at the cost of performance
            if args.agent and args.agent.get('half_cpu', True): jobs //= 2
                
            with pool_type(jobs) as pool:
                for itr, stats_events in enumerate(tqdm_redirect(pool.imap_unordered(engine, iters_range), total=niters,
                                                                 desc='Iterations simulation progress')):
                    # Record sim results
                    net_events[itr] = stats_events
        
        else:
            disable_output = (args.animate != 0)
            with no_std_context(enabled=disable_output):
                for itr in tqdm_redirect(iters_range, desc='Iterations simulation progress', disable=disable_output):
                    print(f'\nRunning iteration {itr}{" during episode " + str(args.agent["episode"]) if args.agent else ""}, with eps = {tr_rate}:')

                    # Reinitialize network + Random first infected at the beginning of each run BUT the first one
                    # This is needed only in sequential processing since in multiprocessing the nets are deepcopied anyway
                    if itr:
                        engine.reinit_net(first_inf_nodes, args.is_dynamic)

                    # Run simulation
                    stats_events = engine(itr)
                    # Record sim results
                    net_events[itr] = stats_events
                    # A situation in which there is NO event can arise when all first infected nodes are orphans, and rem_orphans=True
                    total_inf = stats_events[-1]['totalInfected'] if stats_events else args.first_inf
                    print(f'---> Result: {total_inf} total infected over time. %healthy = {round(1 - total_inf / args.netsize, 3)}')

        # we can either return the events and the last network objects, OR the events and the network state variables that may have been updated here
        # typically, the first option is for serial processing, while the second is for multiprocessing (where state variables are needed for updates)     
        if return_last_net:
            # if NO network distribution was enabled, we can use args (since it is not deepcopied) to track all average degrees of dynamic graphs
            if args.is_dynamic:
                args.k_i = {
                    '0': (args.avg_deg, true_net.avg_degree(use_weights=True))
                }
            return net_events, (true_net, know_net)
        else:
            return net_events, (args.netsize, args.avg_deg, args.overlap, args.overlap_two)

    
class EngineDual(Engine):
    """
    This class extends the Engine abstract class, defining the context-imbued callable to be used for running simulations in parallel over different epidemic seeds.
    EngineDual is used for simulations in which the transmission network is different than the tracing networks. As such, dual and triad topology configurations are supported.
    
    Notes:
        The separation between EngineOne and EngineDual was done for efficiency reasons, avoiding the overhead of checking for the dual indicator variable multiple times.
    """
    def reinit_net(self, first_inf_nodes, dynamic=False):
        """
        Reinitializes the networks for simulation.

        Args:
            first_inf_nodes (list): A list of nodes to start the simulation from.
            dynamic (bool, optional): Whether the network is dynamic. Defaults to False.
        """
        self.true_net.init_for_simulation(first_inf_nodes, dynamic)
        self.know_net.copy_state_from(self.true_net)
     
    def __call__(self, itr, **kwargs):
        """
        Run the simulation for a given iteration ID, used as a seed for the epidemic stochastic process.

        Args:
            itr (int): The epidemic seed iteration.
            **kwargs (dict, optional): Additional keyword arguments to pass to the simulation.
        Returns:
            results (dict): A dictionary containing the results of the simulation.
        """
        # local vars for efficiency
        args = self.args
        args_dict = vars(args)
        dual = args.dual
        separate_traced = args.separate_traced
        sim_agent_based = args.sim_agent_based
        isolate_s = args.isolate_s
        voc_change = args.voc_change
        samp_already_exp = ('dir' in args.sampling_type)
        samp_min_only = ('min' in args.sampling_type)
        order_nid = ('+' in args.sampling_type)
        noncomp_after = args.noncomp_after
        taur = args.taur
        true_net = self.true_net
        inet = true_net.inet
        sim_id = inet + itr
        # Note: know_net may be a utils.ListDelegator of tracing networks
        know_net = self.know_net
        # we need the tracing networks in this form to pass to get_next_event_sample_only_minimum
        tracing_nets, tr_rate = ([know_net], know_net.count_importance) if dual == 1 else (list(know_net), know_net[0].count_importance)
        # the transition objects
        trans_true = self.trans_true
        trans_know = self.trans_know
        # agent to use for control
        agent = None
        is_record_agent = False
        if args.agent:
            from .agent_factory import create_agent
            # the ID of the simulation, the network size, GPU utilization permissions and the logging path are the responsibility of the Simulator
            agent = create_agent(**args.agent, sim_id=sim_id, n_nodes=args.netsize, save_path=args.save_path, gpu=args.control_gpu)
            # can supply a model that is 'shared' across multiprocessing
            if args.shared and hasattr(agent, 'ranking_model'):
                agent.ranking_model = args.shared.get('model', None)
            is_record_agent = args.is_record_agent
            # control starts after this amount of time
            control_day = args.control_after
            # control starts after control_after_inf is surpassed in the population
            control_after_inf = int(args.control_after_inf if args.control_after_inf >= 1 else args.control_after_inf * args.netsize)
            # whether the initial known is taken from the infectious or all infected
            control_initial_infectious = args.control_initial_known >= 0
            control_initial_known = abs(args.control_initial_known)
            # control making up for missing days
            control_makeup_days = args.control_makeup_days
            # boolean to keep track of the first control iteration
            control_first_iter = True
            # vax immunity delay and history
            control_immunity_delay = args.control_immunity_delay
            vax_history = defaultdict(set)
        
        # seed the random for this network id and iteration
        if args.seed is not None:
            seeder = args.seed + sim_id
            random.seed(seeder)
            np.random.seed(seeder)
        # a workaround for setting the infection seed in the iterations rather than beforehand
        if args.infseed and args.infseed < 0:
            # Random first infected across simulations - seed random locally
            first_inf_nodes = random.Random(abs(args.infseed) + sim_id).sample(true_net.nodes, args.first_inf)
            # Change the state of the first_inf_nodes to 'I' to root the simulation
            true_net.change_state(first_inf_nodes, state='I', update=True)
            
        # drawing vars
        draw = args.draw
        draw_iter = args.draw_iter
        animate = args.animate
        draw_plotter = args.draw_config.get('plotter', 'default')
        figsize = args.draw_config.get('figsize', (20, 10) if draw_plotter == 'default' else (10, 10))
        fontsize = args.draw_config.get('title_fontsize', 15)
        output_path = args.draw_config.get('output_path', 'fig/graph')
        
        # If either of the drawing flag is enabled, instantiate drawing figure and axes, and draw initial state
        if draw or draw_iter:
            # import IPython here to avoid dependency if no drawing is performed
            from IPython import display
            if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config), clear=animate)
                if not draw_iter:
                    sleep(.5)
            else:
                if draw_plotter == 'default':
                    fig, ax = plt.subplots(nrows=1, ncols=int(dual) + 1, figsize=figsize, dpi=150)
                    # we make plots for the true network + all the dual networks
                    ax[0].set_title(NETWORK_TITLE, fontsize=fontsize)
                    true_net.draw(seed=args.netseed, show=False, ax=ax[0], **args.draw_config)
                    ax[1].set_title(NETWORK_TITLE_DUAL, fontsize=fontsize)
                    if dual == 2:
                        ax[1].set_title(NETWORK_TITLE_DUAL_TWO, fontsize=fontsize)
                        ax[-1].set_title(NETWORK_TITLE_DUAL_THREE, fontsize=fontsize)
                    know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], model=args.model, **args.draw_config)
                else:
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150)
                    true_net.is_dual = True
                    true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
            
                display.display(fig, clear=animate)

        # simulation objects
        sim_true = true_net.get_simulator(trans_true, already_exp=samp_already_exp, **args_dict)
        sim_know = know_net.get_simulator(trans_know, already_exp=samp_already_exp, **args_dict)
        
        # number of initial infected
        inf = args.first_inf

        # metrics to record simulation summary
        m = {
            'nI': inf,
            'nE': 0,
            'nH': 0,
            'totalInfected': inf,
            'totalInfectious': inf,
            'totalFalseNegative': 0,
            'totalTraced': 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered': 0,
            'totalHospital': 0,
            'totalDeath': 0,
            'tracingEffortRandom': 0,
            'tracingEffortContact': [0],
            # the following metrics are only available when control agents are used, since testing is separated from tracing only in that case
            'totalDetected': 0,
            'testSrc': 0,
            'testSrcUnique': 0,
            'traceSrc': 0,
            'traceSrcUnique': 0,
        }
        
        # iterator needed for dynamical graph updates
        update_iter = voc_update_iter = 1
        # results over time recorded
        result = []
        # whether the network recently received a dynamic edge udpate
        net_changed = True
        
        if sim_agent_based:
            src_inf = defaultdict(int)
            if self.no_exposed:
                to_inf = 'I'
            else:
                to_inf = 'E'
                ab_exptime = np.random.normal(1 / args.eps, 1, len(true_net.node_list)).clip(min=sim_agent_based - 1)
            if args.spontan:
                ab_rectime = np.random.normal(1 / args.gamma, 1, len(true_net.node_list)).clip(min=sim_agent_based - 1)

        # Infinite loop if `args.nevents` not specified (run until no event possible) OR run a number of `args.nevents` otherwise
        if args.nevents > 0:
            events_range = range(args.nevents)
            stop_after_days = 0
            ab_close_cond = not args.is_dynamic
        else:
            events_range = count()
            stop_after_days = abs(args.nevents) - 1
            ab_close_cond = True

        for i in events_range:
            if sim_agent_based:
                current_time = i
                node_states = true_net.node_states
                node_traced = true_net.node_traced
                close = True
                for n1, n2, w in true_net.edges.data('weight', 1.):
                    if node_states[n1].__contains__('I') and not node_traced[n1] and node_states[n2] == 'S' and not(isolate_s and node_traced[n2]):
                        close = False
                        if random.random() <= w * args.beta:
                            src_inf[n1] += 1
                            node_states[n2] = to_inf
                            true_net.node_infected[n2] = True
                            true_net.inf_time[n2] = current_time
                            m[f'n{to_inf}'] += 1
                            m['totalInfected'] += 1
                    elif node_states[n2].__contains__('I') and not node_traced[n2] and node_states[n1] == 'S' and not(isolate_s and node_traced[n1]):
                        close = False
                        if random.random() <= w * args.beta:
                            src_inf[n2] += 1
                            node_states[n1] = to_inf
                            true_net.node_infected[n1] = True
                            true_net.inf_time[n1] = current_time
                            m[f'n{to_inf}'] += 1
                            m['totalInfected'] += 1
                # decrease time till moving from exposed to infectious
                for n in true_net.node_list:
                    if node_states[n] == 'E':
                        close = False
                        if ab_exptime[n] <= 0:
                            node_states[n] = 'I'
                            m['nI'] += 1
                            m['totalInfectious'] += 1
                            m['nE'] -= 1
                        else:
                            ab_exptime[n] -= 1
                    elif args.spontan and node_states[n] == 'I':
                        close = False
                        if ab_rectime[n] <= 0:
                            node_states[n] = 'R'
                            m['totalRecovered'] += 1
                            m['nI'] -= 1
                        else:
                            ab_rectime[n] -= 1
                    # this allows for the simulation to go on indefinitely until all susceptibles are infected
                    elif not ab_close_cond and node_states[n] == 'S':
                        close = False
                if close:
                    break
            else:
                # this option enables T/N states to be separate from the infection states
                if separate_traced:
                
                    # if samp_min_only then we only sample the minimum exponential Exp(sum(lamdas[i] for i in possible_transitions))
                    if samp_min_only:
                        # we combine all possible lambdas from both infection and tracing networks (so pass tracing-related objs here)
                        e = sim_true.get_next_event_sample_only_minimum(trans_know, tracing_nets)

                        if e is None:
                            break

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

                    # Non-compliance with isolation event
                    if e.to == 'N':
                        # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                        if e.node not in true_net.noncomp_time:
                            m['totalNonCompliant'] += 1

                        sim_true.run_trace_event_for_infect_net(e, False)
                        sim_know.run_trace_event_for_trace_net(e, False)

                    # otherwise, normal logic follows if NOT e.to=T event (with update counts on infection network only)
                    elif not is_trace_event:
                        sim_true.run_event(e)
                        # update the time for the sim_know object (thus replacing the need for 'run_event')
                        sim_know.time = e.time
                        # Update event.FROM counts:
                        #   - for models other than covid, leaving 'I' means a decrease in infectious count
                        #   - for covid, leaving 'Ia' or 'Is' means current infectious count decreases
                        if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                            m['nI'] -= 1
                        elif e.fr == 'E':
                            m['nE'] -= 1
                        elif e.fr == 'H':
                            m['nH'] -= 1

                    # if trace event or trace_h selected and hospital event, the node will be counted as traced (if going to H, it will never become noncompliant)
                    if is_trace_event or (args.trace_h and e.to == 'H'):
                        # the update to total traced counts is done only once (ignore if same nid is traced again)
                        if e.node not in true_net.traced_time:
                            m['totalTraced'] += 1
                            # if S -> T then a person has been incorrectly traced
                            if e.fr == 'S': m['totalFalseTraced'] += 1
                            if e.fr == 'E': m['totalExposedTraced'] += 1

                        # for e.to H events, an infection event that updated the infection network will be run
                        # so for args.trace_h == True, we should not update the infection net twice
                        if e.to != 'H':
                            sim_true.run_trace_event_for_infect_net(e, True)
                        sim_know.run_trace_event_for_trace_net(e, True)

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

                # set the current time to the event time               
                current_time = e.time

                # event.TO Updates are common for all models and all parameter settings

                if e.to == 'I':
                    m['nI'] += 1
                    m['totalInfectious'] += 1
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

                # if args.efforts selected, compute efforts for all tracing networks
                if args.efforts:
                    # loop through nodes to collect list of tracing efforts FOR EACH tracing network
                    efforts_for_all = np.atleast_2d(know_net.compute_efforts(taur))
                    # random (testing) effort will be the same for all dual networks - element 0 in each result list      
                    m['tracingEffortRandom'] = efforts_for_all[0, 0]
                    # the contact tracing effort is DIFFERENT across the dual networks - element 1 in each result list
                    m['tracingEffortContact'] = efforts_for_all[:, 1]
                    
            # if there are any nodes that can still impact transmission, then any_inf will be True  
            any_inf = bool(m['nE'] + m['nI'])
            
            # we run the next blocks only when nodes are still active and separate_traced is enabled
            if any_inf and separate_traced:
                ### Hook point for external control measures ! ###
                # for this to execute, an agent config needs to have been supplied as param, `control_after`` days need to have passed, 
                # and a minimum of control_after_inf = i_0 must be present in the network
                if agent and current_time >= control_day and m['totalInfected'] >= control_after_inf:
                    if control_makeup_days == 0:
                        # no corrections
                        missed_days = 0
                        control_day = int(current_time)
                    elif control_makeup_days == 1:
                        # corrections for continuous-time simulations will happen all at once
                        missed_days = int(current_time - control_day)
                        control_day = int(current_time)
                    else:
                        # corrections for continuous-time simulations will happen over time until the true time is caught up
                        missed_days = 0
                        
                    if is_record_agent:
                        agent.control(true_net, control_day, initial_known_ids, net_changed, missed_days)
                    else:
                        node_states = true_net.node_states
                        # in the first iteration, we also assume some knowledge about the infection statuses exists
                        # this knowledge is controlled by args.initial_known
                        if control_first_iter:
                            control_first_iter = False
                            initial_total = np.nonzero(np.char.startswith(node_states, 'I') if control_initial_infectious else np.array(true_net.node_infected))[0]
                            # the agent is given initial information on a subset of the infectious/infected nodes
                            initial_known_len = int(control_initial_known if control_initial_known >= 1 else control_initial_known * len(initial_total))
                            # here, one can randomly sample the known IDs instead of taking the first ones, but on average it should not significantly impact the results
                            # initial_known_ids = np.random.choice(initial_total, size=initial_known_len, replace=False)
                            initial_known_ids = initial_total[:initial_known_len]
                        else:
                            initial_known_ids = ()
                        # ranking of nodes happens here
                        new_pos, traced, vax_history[control_day] = agent.control(know_net, control_day, initial_known_ids, net_changed, missed_days)
                        m['totalDetected'] += len(new_pos)
                        # total traced includes both new positives and contact traced
                        m['totalTraced'] += len(new_pos)
                        for nid in new_pos:
                            # selecting a positive case for retesting should never happen
                            if know_net.node_traced[nid]:
                                raise ValueError('A duplicate positive test was encountered!')
                            state = node_states[nid]
                            event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                            sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                            sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                            if sim_agent_based and nid in src_inf:
                                m['testSrc'] += src_inf[nid]
                                m['testSrcUnique'] += 1                                
                        for nid in traced:
                            # the 'traced' list can contain already-isolated nodes (e.g. part of 'new_pos' OR vaccinations)
                            if not know_net.node_traced[nid]:
                                state = node_states[nid]
                                event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                                sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                                sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                                # update traced count
                                m['totalTraced'] += 1
                                # if S -> T then a person has been incorrectly traced
                                if state == 'S': m['totalFalseTraced'] += 1
                                if state == 'E': m['totalExposedTraced'] += 1
                                if sim_agent_based and nid in src_inf:
                                    m['traceSrc'] += src_inf[nid]
                                    m['traceSrcUnique'] += 1
                        # print(f'Vaccinated at {control_day}', vax_history[control_day])
                        if control_day >= control_immunity_delay:
                            vaxed_immune = vax_history[control_day - control_immunity_delay]
                            # print(control_day, vaxed_immune)
                            for nid in vaxed_immune:
                                if not know_net.node_traced[nid]:
                                    state = node_states[nid]
                                    event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                                    sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                                    sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                                    # update traced count
                                    m['totalTraced'] += 1
                        
                    # increment the control iterations contor
                    control_day += 1
                    # mark that the network nodes + eddges have not changed since this latest control step
                    net_changed = False
                
                # if args.noncomp_after selected, nodes that have been isolating for longer than noncomp_after 
                # automatically become N
                if noncomp_after is not None:
                    node_traced = true_net.node_traced
                    node_states = true_net.node_states
                    traced_time = true_net.traced_time
                    for nid in true_net.node_list:
                        if node_traced[nid] and (current_time - traced_time[nid] >= noncomp_after):
                            event = sim_true.get_event_with_config(node=nid, fr='T', to='N', time=current_time)
                            # remove nid from traced_time to stop it from influencing the tracing chances of its neighbors after exit
                            sim_true.run_trace_event_for_infect_net(event, to_traced=False, legal_isolation_exit=True, order_nid=order_nid)
                            sim_know.run_trace_event_for_trace_net(event, to_traced=False, legal_isolation_exit=True)

            # False Neg: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalseNegative'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])
            
            # draw network at the end of each inner state if option selected
            if draw_iter:
                sleep(int(draw_iter))
                print('State after events iteration ' + str(i) + ':')
                if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                    display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config), clear=animate)
                else:
                    fig = plt.figure(1)
                    clear_axis(ax)
                    if draw_plotter == 'default':
                        ax[0].set_title(NETWORK_TITLE, fontsize=fontsize)
                        true_net.draw(seed=args.netseed, show=False, ax=ax[0], **args.draw_config)
                        ax[1].set_title(NETWORK_TITLE_DUAL, fontsize=fontsize)
                        if dual == 2:
                            ax[1].set_title(NETWORK_TITLE_DUAL_TWO, fontsize=fontsize)
                            ax[-1].set_title(NETWORK_TITLE_DUAL_THREE, fontsize=fontsize)
                        know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], model=args.model, **args.draw_config)
                    else:
                        true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
                    display.display(fig, clear=animate)

            # record metrics after event run for time current_time=e.time
            m['time'] = current_time
            result.append(StatsEvent(**m))
            # close simulation if there are no more possible events on the infection network
            if (not any_inf and not m['nH']) or (stop_after_days > 0 and current_time >= stop_after_days):
                break
                
            # allow for dynamic changes in the viral capabilities
            # only available for COVID, and dependent on voc_after being an integer expressing DAYS required for the update to take effect
            if voc_change and self.is_covid and isinstance(args.voc_after, int) and current_time >= args.voc_after * voc_update_iter:
                trans = sim_true.trans
                if voc_change[0]:
                    # Infections spread based on true_net connections depending on nid
                    beta = args.beta * voc_change[0] ** voc_update_iter
                    add_trans(trans, 'S', 'E', get_stateful_sampling_func('expFactorTimesCountMultiState',
                                                                          states=['Is'], lamda=beta, exp=samp_already_exp,
                                                                          rel_states=['I', 'Ia'], rel=args.rel_beta))
                if len(voc_change) > 1 and voc_change[1]:
                    # probability ph of getting hospitalized scaled with voc_change[0]
                    ph = args.ph * voc_change[1] ** voc_update_iter
                    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
                    hosp_rec = args.gamma * ph
                    hosp_ded = args.gamma * (1 - ph)
                    add_trans(trans, 'Is', 'R', get_stateless_sampling_func(hosp_rec , samp_already_exp), replace=True)
                    add_trans(trans, 'Is', 'H', get_stateless_sampling_func(hosp_ded, samp_already_exp), replace=True)
                if len(voc_change) > 2 and voc_change[2]:
                    # rate of hospitalized to D decaying by a factor of voc_change[1]
                    lambdahd = args.lamdahd * voc_change[2] ** voc_update_iter
                    add_trans(trans, 'H', 'D', get_stateless_sampling_func(lambdahd, samp_already_exp), replace=True)
                voc_update_iter += 1

            # allow for dynamic updates of the network to occur after update_after days IF ANY PROVIDED
            if args.is_dynamic and current_time >= args.update_after * update_iter:
                # IF update_after is given as an int, we interpret updates as happening after full DAYS
                # In this case, the current_time may correspond to an update DAY further ahead than update_after, 
                # and therefore, update_iter needs to be modified accordingly
                if isinstance(args.update_after, int):
                    update_iter = int(current_time // args.update_after)
                try:
                    # at this stage nettype can either be a simple string, in which case we sample the next available edges from the network object
                    # OR it can be a dict containining keys {1, 2, ...} corresponding to update times mapped to dynamic-update sequences
                    edges_to_update = (true_net.sample_edges(update_iter),) \
                                        if isinstance(args.nettype, str) else args.nettype[str(update_iter)]
                    if edges_to_update:
                        inf_update = edges_to_update[0]
                        net_update = False
                        # check if an update for infection network is supplied
                        # this check is agnostic for default iterables and np.arrays
                        if is_not_empty(inf_update):
                            net_update = True
                            initial_nodes = set(true_net.nodes)
                            true_net.clear_edges()
                            # edges_to_update has as first element the update for the infection network
                            true_net.add_links(inf_update, update=False, reindex=args.reindex)
                            new_nodes = true_net.nodes - initial_nodes
                            if new_nodes:
                                # make sure new nodes have a state and traced entry and they are the defaults
                                true_net.add_mult(ids=new_nodes, state='S', traced=False)
                                know_net.add_mult(ids=new_nodes, state='S', traced=False)
                            # update the node_list parameter and rerun the removal of orphans if the flag is set
                            # note that new_nodes need not be filled for new nodes to no longer be orphans (since we have new edges)
                            node_list = list(true_net)
                            # rerun removal of orphans
                            if args.rem_orphans:
                                for nid in true_net.nodes:
                                    if true_net.degree(nid) == 0:
                                        # update active node list
                                        node_list.remove(nid)
                                        # mark orphan as traced
                                        true_net.node_traced[nid] = True
                            # update active node_list without losing the pointer; this will allow tracing nets to see updates
                            true_net.node_list.clear()
                            true_net.node_list.extend(node_list)
                            # update counts WITHOUT traced
                            true_net.update_counts()
                            # mark no node as last updated, since rates need to be updated for all nodes
                            sim_true.last_updated = []
                            net_changed = True
                            # keep track of new average degrees IF no distribution was enabled (making args SHARED)
                            if not args.multip:
                                args.k_i[str(update_iter)] = (true_net.avg_degree(), true_net.avg_degree(use_weights=True))

                        len_edges = len(edges_to_update)
                        # if edges have also been supplied for the dual networks, they will appear starting from index 1
                        if len_edges > 1:
                            # how many iters are executed depend on how many dual networks and how large the list of updates is
                            execute_iters = ceil(dual * (len_edges - 1) / 2) + 1
                            # update if we have any entry for the tracing nets dynamic links
                            for i in range(1, execute_iters):
                                # make sure the edges_to_update[i] entry is not None -> None can be used to skip update for a network
                                update_edges = edges_to_update[i]
                                if is_not_empty(update_edges):
                                    trace_net = tracing_nets[i-1]
                                    trace_net.clear_edges()
                                    trace_net.add_links(edges_to_update[i], update=True, update_with_traced=separate_traced, reindex=args.reindex)
                        # if edges have not been supplied, and there was an infection net update, recalculate link noising of tracing nets
                        elif net_update:
                            # reinitialize the network seed
                            net_seed = random.randint(0, 1e9) if args.netseed is None else args.netseed + inet
                            # Note this will also copy over the states
                            know_net = network.get_dual(true_net, args.overlap, args.zadd, args.zrem, args.uptake, args.maintain_overlap,
                                                        nseed=net_seed+1, inet=inet, count_importance=tr_rate, **args_dict)
                            if dual == 2:
                                know_net_two = network.get_dual(true_net, args.overlap_two, args.zadd_two, args.zrem_two, args.uptake_two, args.maintain_overlap_two, 
                                                                nseed=net_seed+2, inet=inet, count_importance=args.taut_two, **args_dict)
                                # know_net becomes a ListDelegator of the 2 networks
                                know_net = ListDelegator(know_net, know_net_two)

                # If the specified key cannot be found, then there is no dynamic edge update for this time
                except KeyError:
                    pass
                update_iter += 1
                           
        if draw:
            print('Final state:')
            if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config))
            else:
                fig = plt.figure(1)
                clear_axis(ax)
                if draw_plotter == 'default':
                    ax[0].set_title(NETWORK_TITLE, fontsize=fontsize)
                    true_net.draw(show=False, ax=ax[0], **args.draw_config)
                    ax[1].set_title(NETWORK_TITLE_DUAL, fontsize=fontsize)
                    if dual == 2:
                        ax[1].set_title(NETWORK_TITLE_DUAL_TWO, fontsize=fontsize)
                        ax[-1].set_title(NETWORK_TITLE_DUAL_THREE, fontsize=fontsize)
                    know_net.draw(pos=true_net.pos, show=False, ax=ax[1:], model=args.model, **args.draw_config)
                else:
                    true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
                display.display(fig, clear=animate)
                if draw == 2:
                    plt.savefig(f"{output_path}-{sim_id}.pdf", bbox_inches='tight')
            
        if not animate:
            plt.close()

        # call `agent.finish` if the agent exists and it has a `finish` method
        if agent is not None:
            # After an iteration finishes, if a control agent was supplied, it may require some routines to be called here (e.g. for logging)
            try:
                agent.finish(m['totalInfected'], args)
            except (AttributeError, TypeError):
                pass
            # cleanup the agent object
            del agent
        # log the timelag histogram of tracing since infection
        inf_traced_time = set(true_net.inf_time.keys()) & set(true_net.traced_time.keys())
        if result:
            result[-1].timelag_hist = Counter([int(true_net.traced_time[nid] - true_net.inf_time[nid]) for nid in inf_traced_time])
        
        return result

            
class EngineOne(Engine):
    """
    This class extends the Engine abstract class, defining the context-imbued callable to be used for running simulations in parallel over different epidemic seeds.
    EngineOne is used for simulations that are run over a single network, shared between the infection and tracing processes.

    Notes:
        The separation between EngineOne and EngineDual was done for efficiency reasons, avoiding the overhead of checking for the dual indicator variable multiple times.
    """
    def reinit_net(self, first_inf_nodes, dynamic=False):
        """
        Reinitializes the network for simulation.

        Args:
            first_inf_nodes (list): A list of nodes to start the simulation from.
            dynamic (bool, optional): Whether the network is dynamic. Defaults to False.
        """
        self.true_net.init_for_simulation(first_inf_nodes, dynamic)
    
    def __call__(self, itr, **kwargs):
        """
        Run the simulation for a given iteration ID, used as a seed for the epidemic stochastic process.

        Args:
            itr (int): The epidemic seed iteration.
            **kwargs (dict, optional): Additional keyword arguments to pass to the simulation.
        Returns:
            results (dict): A dictionary containing the results of the simulation.
        """
        # local vars for efficiency
        args = self.args
        args_dict = vars(args)
        separate_traced = args.separate_traced
        sim_agent_based = args.sim_agent_based
        voc_change = args.voc_change
        isolate_s = args.isolate_s
        samp_already_exp = ('dir' in args.sampling_type)
        samp_min_only = ('min' in args.sampling_type)
        order_nid = ('+' in args.sampling_type)
        noncomp_after = args.noncomp_after
        taur = args.taur
        # number of initial infected
        inf = args.first_inf
        true_net = self.true_net
        inet = true_net.inet
        sim_id = inet * args.niters + itr
        # the transition object
        trans_true = self.trans_true
        # agent to use for control
        agent = None
        is_record_agent = False

        if args.agent:
            from .agent_factory import create_agent
            # the ID of the simulation, the network size, GPU utilization permissions and the logging path are the responsibility of the Simulator
            agent = create_agent(**args.agent, sim_id=sim_id, n_nodes=args.netsize, save_path=args.save_path, gpu=args.control_gpu)
            # can supply a model that is 'shared' across multiprocessing
            if args.shared and hasattr(agent, 'ranking_model'):
                agent.ranking_model = args.shared.get('model', None)
            is_record_agent = args.is_record_agent
            # control starts after this amount of time
            control_day = args.control_after
            # control starts after control_after_inf is surpassed in the population
            control_after_inf = int(args.control_after_inf if args.control_after_inf >= 1 else args.control_after_inf * args.netsize)
            # whether the initial known is taken from the infectious or all infected
            control_initial_infectious = args.control_initial_known >= 0
            control_initial_known = abs(args.control_initial_known)
            # control making up for missing days
            control_makeup_days = args.control_makeup_days
            # boolean to keep track of the first control iteration
            control_first_iter = True
            # vax immunity delay and history
            control_immunity_delay = args.control_immunity_delay
            vax_history = defaultdict(set)
        
        # seed the random for this network id and iteration
        if args.seed is not None:
            seeder = args.seed + sim_id
            random.seed(seeder)
            np.random.seed(seeder)
        # a workaround for setting the infection seed in the iterations rather than beforehand
        if args.infseed and args.infseed < 0:
            # Random first infected across simulations - seed random locally
            first_inf_nodes = random.Random(abs(args.infseed) + sim_id).sample(true_net.nodes, args.first_inf)
            # Change the state of the first_inf_nodes to 'I' to root the simulation
            true_net.change_state(first_inf_nodes, state='I', update=True)
        
        # drawing vars
        draw = args.draw
        draw_iter = args.draw_iter
        animate = args.animate
        draw_plotter = args.draw_config.get('plotter', 'default')
        figsize = args.draw_config.get('figsize', (20, 10) if draw_plotter == 'default' else (10, 10))
        fontsize = args.draw_config.get('title_fontsize', 15)
        output_path = args.draw_config.get('output_path', 'fig/graph')
        
        # If either of the drawing flag is enabled, instantiate drawing figure and axes, and draw initial state
        if draw or draw_iter > 0:
            # import IPython here to avoid dependency if no drawing is performed
            from IPython import display
            if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config), clear=animate)
                if not draw_iter:
                    sleep(.5)
            else:
                if draw_plotter == 'default':
                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=150)
                    ax[0].set_title('Infection Progress', fontsize=fontsize)
                    true_net.is_dual = False
                    true_net.draw(seed=args.netseed, show=False, ax=ax[0], **args.draw_config)
                    ax[1].set_title('Tracing Progress', fontsize=fontsize)
                    true_net.is_dual = True
                    true_net.draw(show=False, ax=ax[1], model=args.model, **args.draw_config)
                else:
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150)
                    true_net.is_dual = True
                    true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
                    
                display.display(fig, clear=animate)
            
        # set the dual flag such that true_net can be interpreted as the tracing network
        true_net.is_dual = True
        # in the case of separate_traced, mimic dual simulation objects as if dual net was run (sim_true + sim_know)
        if separate_traced:
            # infection simulator in this case should not be able to run 'T' events
            transition_items = defaultdict(list)
            for state in trans_true:
                for transition, func in trans_true[state]:
                    if transition != 'T':
                        transition_items[state].append((transition, func))
            sim_true = true_net.get_simulator(transition_items, already_exp=samp_already_exp, **args_dict)
            # the tracing simulator will calibrate on the first call to only accept 'T' events
            sim_know = true_net.get_simulator(trans_true, already_exp=samp_already_exp, **args_dict)
        else:
            # base simulation object with all transitions possible
            sim_true = sim_know = true_net.get_simulator(trans_true, already_exp=samp_already_exp, **args_dict)

        # metrics to record simulation summary
        m = {
            'nI': inf,
            'nE': 0,
            'nH': 0,
            'totalInfected': inf,
            'totalInfectious': inf,
            'totalFalseNegative': 0,
            'totalTraced': 0,
            'totalFalseTraced': 0,
            'totalExposedTraced': 0,
            'totalNonCompliant': 0,
            'totalRecovered': 0,
            'totalHospital': 0,
            'totalDeath': 0,
            'tracingEffortRandom': 0,
            'tracingEffortContact': [0],
            # next metrics are only available when control agents are used
            'totalDetected': 0,
            'testSrc': 0,
            'testSrcUnique': 0,
            'traceSrc': 0,
            'traceSrcUnique': 0,
        }

        # iterator needed for dynamical graph updates (these happen after every args.update_after period of time has elapsed)
        update_iter = voc_update_iter = 1
        # results over time recorded
        result = []
        # whether the network recently received a dynamic edge udpate
        net_changed = True
        
        if sim_agent_based:
            src_inf = defaultdict(int)
            if self.no_exposed:
                to_inf = 'I'
            else:
                to_inf = 'E'
                ab_exptime = np.random.normal(1 / args.eps, 1, len(true_net.node_list)).clip(min=sim_agent_based - 1)
            if args.spontan:
                ab_rectime = np.random.normal(1 / args.gamma, 1, len(true_net.node_list)).clip(min=sim_agent_based - 1)

        # Infinite loop if `args.nevents` not specified (run until no event possible) OR run a number of `args.nevents` otherwise
        if args.nevents > 0:
            events_range = range(args.nevents)
            stop_after_days = 0
            ab_close_cond = not args.is_dynamic
        else:
            events_range = count()
            stop_after_days = abs(args.nevents) - 1
            ab_close_cond = True

        for i in events_range:
            if sim_agent_based:
                current_time = i
                node_states = true_net.node_states
                node_traced = true_net.node_traced
                close = True
                for n1, n2, w in true_net.edges.data('weight', 1.):
                    if node_states[n1].__contains__('I') and not node_traced[n1] and node_states[n2] == 'S' and not(isolate_s and node_traced[n2]):
                        close = False
                        if random.random() <= w * args.beta:
                            src_inf[n1] += 1
                            node_states[n2] = to_inf
                            true_net.node_infected[n2] = True
                            true_net.inf_time[n2] = current_time
                            m[f'n{to_inf}'] += 1
                            m['totalInfected'] += 1
                    elif node_states[n2].__contains__('I') and not node_traced[n2] and node_states[n1] == 'S' and not(isolate_s and node_traced[n1]):
                        close = False
                        if random.random() <= w * args.beta:
                            src_inf[n2] += 1
                            node_states[n1] = to_inf
                            true_net.node_infected[n1] = True
                            true_net.inf_time[n1] = current_time
                            m[f'n{to_inf}'] += 1
                            m['totalInfected'] += 1
                # decrease time till moving from exposed to infectious
                for n in true_net.node_list:
                    if node_states[n] == 'E':
                        close = False
                        if ab_exptime[n] <= 0:
                            node_states[n] = 'I'
                            m['nI'] += 1
                            m['totalInfectious'] += 1
                            m['nE'] -= 1
                        else:
                            ab_exptime[n] -= 1
                    elif args.spontan and node_states[n] == 'I':
                        close = False
                        if ab_rectime[n] <= 0:
                            node_states[n] = 'R'
                            m['totalRecovered'] += 1
                            m['nI'] -= 1
                        else:
                            ab_rectime[n] -= 1
                    # this allows for the simulation to go on indefinitely until all susceptibles are infected
                    elif not ab_close_cond and node_states[n] == 'S':
                        close = False
                if close:
                    break
            else:
                if separate_traced:
                    # if samp_min_only then we only sample the minimum exponential Exp(sum(lamdas[i] for i in possible_transitions))
                    if samp_min_only:
                        # we combine all possible lambdas from both infection and tracing
                        # in this case we do not have dual networks, so all will be performed on top of the true net
                        e = sim_true.get_next_event_sample_only_minimum(trans_true, [true_net])

                        if e is None:
                            break

                    # otherwise, all exponentials are sampled as normal (either they are already exp or will be sampled later)
                    else:
                        e1 = sim_true.get_next_event()
                        # allow for separate tracing events to also be considered in the current events loop
                        e2 = sim_know.get_next_trace_event()

                        # If no more events left, break out of the loop
                        if e1 is None and e2 is None:
                            break

                        e = e1 if (e2 is None or (e1 is not None and e1.time < e2.time)) else e2

                    # if the event chosen is a tracing event, separate logic follows (NOT updating event.FROM counts!)
                    is_trace_event = (e.to == 'T')

                    # Non-compliance with isolation event
                    if e.to == 'N':
                        # the update to total noncompliant counts is done only once (ignore if same nid is noncompliant again)
                        if e.node not in true_net.noncomp_time:
                            m['totalNonCompliant'] += 1

                        sim_true.run_trace_event_for_infect_net(e, False)
                        sim_know.run_trace_event_for_trace_net(e, False)

                    # otherwise, normal logic follows if NOT e.to=T event (with update counts on infection network only)
                    elif not is_trace_event:
                        sim_true.run_event(e)
                        # update the time for the 'fake' sim_know object (thus replacing the need for 'run_event')
                        sim_know.time = e.time

                        # Update event.FROM counts:
                        #   - for models other than covid, leaving 'I' means a decrease in infectious count
                        #   - for covid, leaving 'Ia' or 'Is' means current infectious count decreases
                        if (not self.is_covid and e.fr == 'I') or e.fr == 'Ia' or e.fr == 'Is':
                            m['nI'] -= 1
                        elif e.fr == 'E':
                            m['nE'] -= 1
                        elif e.fr == 'H':
                            m['nH'] -= 1
                            
                    # if trace event or trace_h selected and hospital event, the node will be counted as traced (if going to H, it will never become noncompliant)
                    if is_trace_event or (args.trace_h and e.to == 'H'):
                        # the update to total traced counts is done only once (ignore if same nid is traced again)
                        if e.node not in true_net.traced_time:
                            m['totalTraced'] += 1
                            # if S -> T then a person has been incorrectly traced
                            if e.fr == 'S': m['totalFalseTraced'] += 1
                            if e.fr == 'E': m['totalExposedTraced'] += 1

                        # for e.to H events, an infection event that updated the infection network will be run
                        # so for args.trace_h == True, we should not update the infection net twice
                        if e.to != 'H':
                            sim_true.run_trace_event_for_infect_net(e, True)
                        sim_know.run_trace_event_for_trace_net(e, True)

                else:
                    e = sim_true.get_next_event()
                    # If no more events left, break out of the loop
                    if e is None:
                        break

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

                # set the current time to the event time
                current_time = e.time

                # event.TO Updates are common for all models

                if e.to == 'I':
                    m['nI'] += 1
                    m['totalInfectious'] += 1
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

                # if args.efforts selected, compute efforts for all tracing networks
                if args.efforts:
                    # loop through nodes to collect list of tracing efforts FOR EACH tracing network
                    efforts_for_all = true_net.compute_efforts(taur)
                    # random (testing) effort is the first element (no dual networks here)
                    m['tracingEffortRandom'] = efforts_for_all[0]
                    # the contact tracing effort is the second (no dual networks here)
                    m['tracingEffortContact'] = [efforts_for_all[1]]
            
            # if there are any nodes that can still impact transmission, then any_inf will be True  
            any_inf = bool(m['nE'] + m['nI'])
            
            # we run the next blocks only when nodes are still active and separate_traced is enabled
            if any_inf and separate_traced:
                ### Hook point for external control measures ###
                # for this to execute, an agent config needs to be supplied as param, control_after = t_0 days must have passed, 
                # and a minimum of control_after_inf = i_0 must be present in the simulation
                if agent and current_time >= control_day and m['totalInfected'] >= control_after_inf:
                    if control_makeup_days == 0:
                        # no corrections
                        missed_days = 0
                        control_day = int(current_time)
                    elif control_makeup_days == 1:
                        # corrections for continuous-time simulations will happen all at once
                        missed_days = int(current_time - control_day)
                        control_day = int(current_time)
                    else:
                        # corrections for continuous-time simulations will happen over time until the true time is caught up
                        missed_days = 0
                        
                    if is_record_agent:
                        agent.control(true_net, control_day, initial_known_ids, net_changed, missed_days)
                    else:
                        node_states = true_net.node_states
                        # in the first iteration, we also assume some knowledge about the infection statuses exists
                        # this knowledge is controlled by args.initial_known
                        if control_first_iter:
                            control_first_iter = False
                            initial_total = np.nonzero(np.char.startswith(node_states, 'I') if control_initial_infectious else np.array(true_net.node_infected))[0]
                            # the agent is given initial information on a subset of the infectious/infected nodes
                            initial_known_len = int(control_initial_known if control_initial_known >= 1 else control_initial_known * len(initial_total))
                            initial_known_ids = initial_total[:initial_known_len]
                            # initial_known_ids = np.random.choice(initial_total, size=initial_known_len, replace=False) - gets a random sample instead
                        else:
                            initial_known_ids = ()
                        # ranking of nodes happens here
                        new_pos, traced, vax_history[control_day] = agent.control(true_net, control_day, initial_known_ids, net_changed, missed_days)
                        m['totalDetected'] += len(new_pos)
                        # total traced includes both new positives and contact traced
                        m['totalTraced'] += len(new_pos)
                        for nid in new_pos:
                            # selecting a positive case for retesting should never happen
                            if true_net.node_traced[nid]:
                                raise ValueError('A duplicate positive test was encountered!')
                            state = node_states[nid]
                            event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                            sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                            sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                            if sim_agent_based and nid in src_inf:
                                m['testSrc'] += src_inf[nid]
                                m['testSrcUnique'] += 1                                
                        for nid in traced:
                            # the 'traced' list can contain already-isolated nodes (e.g. part of 'new_pos' OR vaccinations)
                            if not true_net.node_traced[nid]:
                                state = node_states[nid]
                                event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                                sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                                sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                                
                                # update traced count
                                m['totalTraced'] += 1
                                # if S -> T then a person has been incorrectly traced
                                if state == 'S': m['totalFalseTraced'] += 1
                                if state == 'E': m['totalExposedTraced'] += 1
                                if sim_agent_based and nid in src_inf:
                                    m['traceSrc'] += src_inf[nid]
                                    m['traceSrcUnique'] += 1
                        # print(f'Vaccinated at {control_day}', vax_history[control_day])
                        if control_day >= control_immunity_delay:
                            vaxed_immune = vax_history[control_day - control_immunity_delay]
                            # print(control_day, vaxed_immune)
                            for nid in vaxed_immune:
                                if not true_net.node_traced[nid]:
                                    state = node_states[nid]
                                    event = sim_true.get_event_with_config(node=nid, fr=state, to='T', time=current_time)
                                    sim_true.run_trace_event_for_infect_net(event, to_traced=True, order_nid=order_nid)
                                    sim_know.run_trace_event_for_trace_net(event, to_traced=True)
                                    # update traced count
                                    m['totalTraced'] += 1
                        
                    # increment the control iterations contor
                    control_day += 1
                    # mark that the network nodes + eddges have not changed since this latest control step
                    net_changed = False
                
                # if args.noncomp_after selected, nodes that have been isolating for longer than noncomp_after 
                # automatically become N
                if noncomp_after is not None:
                    node_traced = true_net.node_traced
                    node_states = true_net.node_states
                    traced_time = true_net.traced_time
                    for nid in true_net.node_list:
                        if node_traced[nid] and (current_time - traced_time[nid] >= noncomp_after):
                            event = sim_true.get_event_with_config(node=nid, fr='T', to='N', time=current_time)
                            # remove nid from traced_time to stop it from influencing the tracing chances of its neighbors after exit
                            sim_true.run_trace_event_for_infect_net(event, to_traced=False, legal_isolation_exit=True, order_nid=order_nid)
                            sim_know.run_trace_event_for_trace_net(event, to_traced=False, legal_isolation_exit=True)
                
            # False Neg: Infectious but not Traced
            # infectious - traced_infectious = infectious - (traced - traced_false - traced_exposed)
            m['totalFalseNegative'] = m['totalInfectious'] - (m['totalTraced'] - m['totalFalseTraced'] - m['totalExposedTraced'])
            
            if draw_iter:
                sleep(int(draw_iter))
                print('State after events iteration ' + str(i) + ':')
                if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                    display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config), clear=animate)
                else:
                    fig = plt.figure(1)
                    clear_axis(ax)
                    if draw_plotter == 'default':
                        ax[0].set_title('Infection Progress', fontsize=fontsize)
                        true_net.is_dual = False
                        true_net.draw(seed=args.netseed, show=False, ax=ax[0], **args.draw_config)
                        ax[1].set_title('Tracing Progress', fontsize=fontsize)
                        true_net.is_dual = True
                        true_net.draw(show=False, ax=ax[1], model=args.model, **args.draw_config)
                    else:
                        true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
                    display.display(fig, clear=animate)
                
            # record metrics after event run for time current_time=e.time
            m['time'] = current_time
            result.append(StatsEvent(**m))
            # close simulation if there are no more possible events on the infection network
            if (not any_inf and not m['nH']) or (stop_after_days > 0 and current_time >= stop_after_days):
                break
            
            # allow for dynamic changes in the viral capabilities
            # only available for COVID, and dependent on voc_after being an integer expressing DAYS required for the update to take effect
            if voc_change and self.is_covid and isinstance(args.voc_after, int) and current_time >= args.voc_after * voc_update_iter:
                trans = sim_true.trans
                if voc_change[0]:
                    # Infections spread based on true_net connections depending on nid
                    beta = args.beta * voc_change[0] ** voc_update_iter
                    add_trans(trans, 'S', 'E', get_stateful_sampling_func('expFactorTimesCountMultiState',
                                                                               states=['Is'], lamda=beta, exp=samp_already_exp,
                                                                               rel_states=['I', 'Ia'], rel=args.rel_beta))
                if len(voc_change) > 1 and voc_change[1]:
                    # probability ph of getting hospitalized scaled with voc_change[0]
                    ph = args.ph * voc_change[1] ** voc_update_iter
                    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
                    hosp_rec = args.gamma * ph
                    hosp_ded = args.gamma * (1 - ph)
                    add_trans(trans, 'Is', 'R', get_stateless_sampling_func(hosp_rec , samp_already_exp), replace=True)
                    add_trans(trans, 'Is', 'H', get_stateless_sampling_func(hosp_ded, samp_already_exp), replace=True)
                if len(voc_change) > 2 and voc_change[2]:
                    # rate of hospitalized to D decaying by a factor of voc_change[1]
                    lambdahd = args.lamdahd * voc_change[2] ** voc_update_iter
                    add_trans(trans, 'H', 'D', get_stateless_sampling_func(lambdahd, samp_already_exp), replace=True)
                voc_update_iter += 1
                
            # allow for dynamic updates of the network to occur after update_after days IF ANY PROVIDED
            # Note: if no edge_sample_size supplied for static networks, is_dynamic will always be False
            if args.is_dynamic and current_time >= args.update_after * update_iter:
                # IF update_after is given as an Integer, we interpret updates as happening after full DAYS
                # In this case, the current_time may correspond to an update DAY further ahead than update_after, 
                # therefore we need to modify update_iter accordingly
                if isinstance(args.update_after, int):
                    update_iter = int(current_time // args.update_after)
                try:
                    # at this stage nettype can either be a simple string, in which case we sample the next available edges from the network object
                    # OR it can be a dict containining keys {1, 2, ...} corresponding to update times mapped to dynamic-update sequences
                    edges_to_update = (true_net.sample_edges(update_iter),) \
                                        if isinstance(args.nettype, str) else args.nettype[str(update_iter)]
                    if edges_to_update:
                        inf_update = edges_to_update[0]
                        # check if an update for infection network is supplied
                        # this check is agnostic for default iterables and np.arrays
                        if is_not_empty(inf_update):
                            initial_nodes = set(true_net.nodes)
                            true_net.clear_edges()
                            # edges_to_update has as first element the update for the infection network
                            true_net.add_links(inf_update, update=False, reindex=args.reindex)
                            new_nodes = true_net.nodes - initial_nodes
                            if new_nodes:
                                # make sure new nodes have a state and traced entry and they are the defaults
                                true_net.add_mult(ids=new_nodes, state='S', traced=False)
                            # update the node_list parameter and rerun the removal of orphans if the flag is set
                            # note that new_nodes need not be filled for new nodes to no longer be orphans (since we have new edges)
                            node_list = list(true_net)
                            # rerun removal of orphans
                            if args.rem_orphans:
                                for nid in true_net.nodes:
                                    if true_net.degree(nid) == 0:
                                        # update active node list
                                        node_list.remove(nid)
                                        # mark orphan as traced
                                        true_net.node_traced[nid] = True
                            # update active node_list without losing the pointer
                            true_net.node_list.clear()
                            true_net.node_list.extend(node_list)
                            # update counts with/without traced
                            true_net.update_counts_with_traced() if separate_traced else true_net.update_counts()
                            # mark no node as last updated, since rates need to be updated for all nodes
                            sim_true.last_updated = []
                            net_changed = True
                            # keep track of new average degrees IF no distribution was enabled (making args SHARED)
                            if not args.multip:
                                args.k_i[str(update_iter)] = (true_net.avg_degree(), true_net.avg_degree(use_weights=True))
                
                # If the specified key cannot be found, then there is no dynamic edge update for this time id
                except KeyError:
                    pass
                update_iter += 1
            
        if draw:
            print('Final state:')
            if 'pyvis' in draw_plotter or 'plotly' in draw_plotter:
                display.display(true_net.draw(seed=args.netseed, model=args.model, **args.draw_config))
            else:
                fig = plt.figure(1)
                clear_axis(ax)
                if draw_plotter == 'default':
                    ax[0].set_title('Infection Progress', fontsize=fontsize)
                    true_net.is_dual = False
                    true_net.draw(show=False, ax=ax[0], **args.draw_config)
                    ax[1].set_title('Tracing Progress', fontsize=fontsize)
                    true_net.is_dual = True
                    true_net.draw(show=False, ax=ax[1], model=args.model, **args.draw_config)
                else:
                    true_net.draw(show=False, ax=ax, seed=args.netseed, model=args.model, **args.draw_config)
                display.display(fig, clear=animate)      
                if draw == 2:
                    plt.savefig(f"{output_path}-{sim_id}.pdf", bbox_inches='tight')
         
        if not animate:
            plt.close()

        # call `agent.finish` if the agent exists and it has a `finish` method
        if agent is not None:
            # After an iteration finishes, if a control agent was supplied, it may require some routines to be called here (e.g. for logging)
            try:
                agent.finish(m['totalInfected'], args)
            except (AttributeError, TypeError):
                pass
            # cleanup the agent object
            del agent
        # log the timelag histogram of tracing since infection
        inf_traced_time = set(true_net.inf_time.keys()) & set(true_net.traced_time.keys())
        if result:
            result[-1].timelag_hist = Counter([int(true_net.traced_time[nid] - true_net.inf_time[nid]) for nid in inf_traced_time])
        
        return result


def clear_axis(ax):
    """
    Clears the given axis or list of axes.

    Args:
        ax (matplotlib.axis.Axis or list): The axis or list of axes to clear.

    Returns:
        None
    """
    try:
        ax.clear()
    except AttributeError:
        for axis in ax:
            axis.clear()