import os
import argparse
import random
import pickle
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable
from math import ceil
from datetime import datetime
from multiprocessing import cpu_count, Manager

from .tracing import utils as ut
from .tracing.stats import StatsProcessor
from .tracing.models import get_transitions_for_model, add_trans
from .tracing.multirun_engine import EngineNet
from .tracing.data_utils import DataLoader


# Covid infection-related parameters according to DiDomenico et al. 2020
# https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-020-01698-4#additional-information
PARAMETER_DEFAULTS = {
    #### Network related parameters:
    ## network wiring type; this can be one of the following: 
    # - a STR with the name of a random graph model (e.g. random, barabasi, ws, powerlaw-cluster) OR a predefined dataset name (SocialEvol)
    # - an Iterable of edges (including nx.Graph objects)
    # - a DICT mapping integer timestamps (0,1,...) to LISTS of predefined networks - infection, tracing #1, tracing #2
    # for DICT the 'update_after' param dictates when each integer key becomes 'active', thus changing the network wiring (dynamic)
    'nettype': 'random',
    ## net size, avg degree, rewire prob, edge weighting switch for various graph types - these only have EFFECT IF nettype is a known nettype (STR type)
    'netsize': 100, 'k': 10., 'p': .05, 'weighted': False,
    # whether to reindex (starting from 0) the nids received as input from an external nettype or not
    'reindex': False,
    ## controls whether edge weights matter for transmission and their normalising factor (only has effect when networks with edges have been provided)
    'use_weights': False, 'K_factor': 10,  # if K_factor = 0, no normalization of weights happens
    ## 0 - tracing happens on same net as infection, 1 - one dual net for tracing, 2 - two dual nets for tracing
    'dual': 1,
    ## overlaps for tracing nets (second is used only if dual == 2)
    'overlap': .8, 'overlap_two': .5,
    ## if no overlap given, tracing net has zadd edges added on random, and zrem removed
    'zadd': 0, 'zrem': 5, # if overlap is set, bool(zadd) is still used to infer whether we want to add random edges or only remove
    ## similar as before but for dual == 2 and if no overlap_two given
    'zadd_two': 0, 'zrem_two': 5, # zadd_two fully replicates the functionality of zadd for the second dual net
    ## maximum percentage of nodes with at least 1 link (adoption rate)
    'uptake': 1., 'uptake_two': 1.,
    ## if maintain_overlap True, then the generator will try to accommodate both the uptake and the overlap
    'maintain_overlap': False, 'maintain_overlap_two': True,
    ## update network after X days
    'update_after': -1,
    ## num of sampling edges (or uniform distrib limits for this num) to be used for a known nettype provided as STR
    'edge_sample_size': [],
    ## controls type of edges subset sampling
    'edge_prob_sampling': True,

    #### Compartmental model parameters:
    ## can be sir, seir or covid
    'model': 'sir',
    ## number of nodes infected at the start of sim
    'first_inf': 1.,
    ## whether to have the Traced status separate from the infection states
    # Note if this is disabled, much of the functionality surrounding noncompliance and self-isolation exit will not work
    'separate_traced': True,
    ## whether Susceptible people are isolated 
    # if True and 'traced', they can't get infected unless noncompliant
    'isolate_s': True,
    ## whether to mark hospitalized as traced
    'trace_h': True,
    ## if -1/None a node can become traced again only after exiting self-isolation
    # not -1/None a node can become traced again after this amount of time (in days) has elapsed after becoming ilegally N
    'trace_after': 7,
    ## percentage of contact reduction effect from possible edges, mask wearing and stay-at-home people
    'contact_reduction': 0, 'mask_uptake': 0., 'sah_uptake': 0.,

    #### Disease-specific parameters that are common for multiple models:
    ## transmission rate -> For Covid, 0.0791 correponding to R0=3.18
    'beta': 0.0791,
    ## latency -> For Covid 3.7 days
    'eps': 1/3.7,
    ## global (spontaneus) recovey rate -> For Covid 2.3 days
    'gamma': 1/2.3,
    ## allow spontaneus recovery (for SIR and SEIR only, for Covid always true)
    'spontan': False,
    ## recovery rate for traced people (if 0, global gamma is used)
    'gammatau': 0,
    ## masks wearing transmission cut
    'mask_cut': .5,

    #### COVID model specific parameters:
    ## probability of being asymptomatic (could also be 0.5)
    'pa': 0.2,
    ## relative infectiousness of Ip/Ia compared to Is (Imperial paper + Medrxiv paper)
    'rel_beta': .5,
    ## relative random tracing (testing) rate of Ia compared to Is
    'rel_taur': .8,
    ## duration of prodromal phase
    'miup': 1/1.5,
    ## probability of being hospitalized (i.e. having severe symptoms Pss) based on age category
    'ph': [0, 0.1, 0.2],
    ## If hospitalized, daily rate entering in R based on age category: children, adults, seniors
    'lamdahr': [0, .083, .033],
    ## If hospitalized, daily rate entering in D based on age category: children, adults, seniors
    'lamdahd': [0, .0031, .0155],
    ## Age-group; Can be 0 - children, 1 - adults, 2 - senior
    'group': 1,
    ## Age-group percentages (to be used when no group is selected, i.e. takes a value of -1/None)
    'group_percent': [.173, .641, .186],
    ## List of 3 elements that encode a change in viral variant:
    # first element is factor change in the transmission rate; 
    # second is factor change in the probability of hospitalization;
    # third element is factor change in the death rate
    'voc_change': [],
    ## time after which the hospitalization/death rates change due to another variant
    # this should be an integer designating full days (it will be ignored if float)
    'voc_after': 15,

    #### Tracing parameters:
    ## testing (random tracing) rate
    'taur': 0.,
    ## contact-tracing rates for first & second tracing networks (if exists)
    'taut': 0.1, 'taut_two': -1.,
    ## number of days of delay on second tracing network compared to first one
    # this is taken into account only if taut_two==-1/None
    'delay_two': 2.,
    ## noncompliance rate; Note this ONLY works for args.separate_traced=True
    # each day the chance of going out of isolation increases by x%
    'noncomp': 0.,
    ## whether the noncomp rate gets multiplied by time difference t_current - t_trace
    'noncomp_dependtime': True,
    ## period after which T becomes automatically N (nonisolating); Note this ONLY works for args.separate_traced=True
    # -1/None means disabled; 14 is standard quarantine
    'noncomp_after': -1,

    #### Simulation controlling parameters
    ## running number of nets, iterations per net and events for each
    # if nevents=0, run until no more events
    'nnets': 1, 'niters': 1, 'nevents': 0,
    ## seed of simulation exponentials; the seed for network initializations; and the first infected seed
    # if -1, both seed and netseed default to None, whereas infseed is ignored (and netseed gets used for sampling the first infected)
    # except for infseed (and only in the case of a positive value), neither seed gets used directly, rather they get incremented by iterators
    'seed': -1, 'netseed': -1, 'infseed': -1,
    ## 0 - no multiprocess, 1 - multiprocess nets, 2 - multiprocess iters, 3 - multiprocess nets and iters (half-half cpus)
    'multip': 0,
    ## sampling procedure for events; can be either of these:
    # dir: the exponential is sampled DIRECTLY from the function registered on the transition object
    # each: the transition obj registers only the lambda rates, the exponential is sampled FOR EACH lambda with exp_sampler.py
    # min: Gillespie's algorithm; the transition obj registers the lambda rates, ONLY the MINIMUM exponential is sampled based on sum
    'sampling_type': 'min',
    ## number of stateless exponential presamples (if -1/None/0, no presampling)
    'presample': -1,
    ## whether or not to remove orphans from the infection network (they will not move state on the infection net)
    'rem_orphans': False,
    ## if rem_orphans, noising of links can be calculated either based on the full node list or the active nonorphan list through this param
    'active_based_noising': False,
    ## wehther efforts are to be computed for each type of tracing (random+contact)
    'efforts': False,
    # agent-based simulation; if 0 EBM is run, otherwise ABM is run with minimal latent transition (sim_agent_based - 1) steps
    'sim_agent_based': 0,

    #### Summary/logging-related parameters
    ## controls how the program returns and what it prints at the end of the simulations
    # -1/None -> always return None, summary never called; 0 -> return summary, no printing; 
    # 1 -> return None, print json summary to stdout; 2 -> return None, print to file; >2 -> return None, print to file with `seed` in name
    'summary_print': -1.,
    ## how many time splits to use for the epidemic summary (if 0, no summary is computed)
    'summary_splits': 30,
    ## number of days for Reff calculation
    'r_window': 7,
    ## first_inf + earlystop_margin determines if a simulation is regarded as early stopped
    'earlystop_margin': 0,
    ## used to calculate fraction of 'contained' epidemics (i.e. overall infected nodes smaller than fraction `alpha_contain`)
    'alpha_contain': 0.4,
    ## whether alternative averages which have no early stopped iterations are to be computed
    'avg_without_earlystop': False,
    ## an id for the experiment to be used by the logging folder structure
    # if ends with '/', the date will be appended to the end of the path
    'exp_id': 'default_id',

    #### Drawing- and printing-related parameters:
    ## controls printing information during the simulation and animation of the infection progress
    # -1 - args printed; 0 - full printing, 1 - print nothing, FORCE animate, >1 - no print, no animation unless `draw`!=0 supplied
    'animate': 0,
    ## 0 - no draw, 1 - draw at start/finish, 2 - draw and save figure at finish
    'draw': 0,
    ## if not 0, draw after each iteration and sleep for this long
    'draw_iter': 0.,
    ## drawing configuration: engine, layout, custom behavior for dual, labels etc.
    'draw_config': {'plotter': 'default', 'layout': 'spring', 'legend': 0, 'with_labels': True,
                    'output_path': 'fig/graph'},
    
    #### Control-related parameters
    ## dict that is utilized to initialize a testing/tracing Agent
    'agent': {},
    ## learning model for ranking nodes
    'ranker': {},
    ## interface with a shared memory object across processes (copied over)
    'shared': None,
    ## number of episodes, eps_start and eps_decay
    'control_schedule': [],
    ## number of infected (or percent of the total infected) that the agent 'knows' about at the start of control; if <0, taken from infectious nodes only
    'control_initial_known': .25,
    ## number of days that need to pass before the control routine is activated
    'control_after': 5,
    ## number of infected (or percent of the total population) that needs to be surpassed until the control routine is activated
    'control_after_inf': .05,
    ## corrections for CT simulations: 0 never, 1 all at once, 2 allow the controller to catch-up dynamically if unspent budget
    'control_makeup_days': 2,
    ## vax immunity delay
    'control_immunity_delay': 21,
    ## whether control uses GPUs
    'control_gpu': False,
}


RECORD_PATH = 'tracing_runs/'
TEMP_PATH = 'temp/'

    
def main(args=None):
    """
    Runs the main logic of the program, given the supplied namespace arguments or command-line arguments if `args` is None.

    Args:
        args: An argparse object containing the arguments. Defaults to None, in which case the command-line arguments are parsed.

    Returns:
        None.
    """
    # Parse the command-line arguments if None were provided
    if args is None:
        args = parse_args()

    time = datetime.now().strftime('%Y-%m-%d--%H-%M/')
    # get env variables and set date information + logging location for the run
    args.cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    # construct the save path for the experiment
    args.save_path = f'{RECORD_PATH}{args.exp_id}'
    # if `exp_id` does not end with '/', this signals that the user wants to use a nested folder structure, with the date as the last folder 
    if not args.exp_id.endswith('/'):
        args.save_path += f'/{time}'
    args.draw_config['output_path'] = f"{args.save_path}/{args.draw_config['output_path']}"
    if not args.animate:
        print(f'\nExperiment date: {time}')
    
    ### First part of code is concerned with overwriting in a sensible manner args arguments, based on incomaptibilities and other criteria
    
    assert isinstance(args.nettype, Iterable), 'The nettype parameter needs to be an Iterable'
    
    # this is a hook point to supply own logic for loading custom datasets (invoked via special strings defined here)
    if isinstance(args.nettype, str):
        ds_path = f'{TEMP_PATH}{args.nettype.replace(":", "_")}'
        try:
            with open(ds_path, 'rb') as f:
                args.nettype = pickle.load(f)
        except (IOError, ImportError, AttributeError, IndexError):
            if 'socialevol' in args.nettype:
                # allow update_after (used to indicate when dynamic edge updates happen) to influence the aggregation 
                # of data (either weekly or daily)
                agg = 'W' if args.update_after > 3 else 'D'
                # proximity probability filtering can be specified after ':'
                options = args.nettype.split(':')
                # defaults
                proxim_prob = None
                time_fr, time_to = '2009-01-05', '2009-06-14'
                use_week_index = False
                try:
                    proxim_prob = float(options[1])
                except (IndexError, ValueError):
                    pass
                try:
                    time_options = options[2].split(',')
                    time_fr = int(time_options[0])
                    time_to = int(time_options[1])
                    use_week_index = (time_options[2].lower() == 'w')
                except (IndexError, ValueError):
                    pass
                # this effectively reads out the data file and does initial filtering and joining
                loader = DataLoader(agg=agg, proxim_prob=proxim_prob)
                # supply to the args.nettype a dictionary, with configuration keys being 'nid', 'Wi', 'Wt' (if which_tr>=0), 
                # and time keys '0', '1', '2' corresponding to the timestamp of the specific dynamical edge update
                # note that the code also supports custom tracing networks (in this SocialEvol example, choose which_tr>=0 for this)
                args.nettype = loader.get_edge_data_for_time(which_inf=0, which_tr=None, time_to=time_to, 
                                                             time_fr=time_fr, use_week_index=use_week_index)
                os.makedirs(TEMP_PATH, exist_ok=True)
                with open(ds_path, 'wb') as f:
                    pickle.dump(args.nettype, f)
            
    # an Iterable that is not a DICT or a STR may be supplied to nettype, in which case we adapt it to the convention DICT(STR -> LIST(nets))
    elif not isinstance(args.nettype, dict):
        args.nettype = {'0': (args.nettype,)} 
        
    # the following step ensures that unselected (-1) parameters are turned to None
    args_dict = vars(args)
    for argkey, argvalue in args_dict.items():
        if argvalue == -1: args_dict[argkey] = None
        
    # default the number of first infected nodes to None, and assume for now we know the nodes of the inf network
    is_nids_known = True
    first_inf_nodes = None
    # one can exclude dynamic edge updates if args.update_after isn't set at all
    args.is_dynamic = args.update_after is not None
    # reflect in the netsize and the is_dynamic param the fact that we may have supplied a predefined infection network through args.nettype, 
    # or a predefined network was loaded in through a special STR supplied through this parameter (in both cases, nettype should be a DICT by this point)
    if isinstance(args.nettype, dict) and '0' in args.nettype:
        try:
            nodes = args.nettype['nid']
            # if nids supplied, we can infer the netsize and the average degree of the network
            args.netsize = len(nodes)
            args.k = 2 * len(args.nettype['0'][0]) / args.netsize
        except (KeyError, IndexError):
            # if nids or edges for time 0 of inf net not supplied, we assume we do not know the netsize or the degree yet
            args.netsize = args.k = -1
            is_nids_known = False
        # the graph is dynamic if any update excepting '0' exists, and the option is turned on through update_after
        args.is_dynamic &= len(args.nettype.keys()) > 1
    # else branch is for random networks, and the nids will be generated based on the range of args.netsize
    else:
        nodes = range(args.netsize)
        # need to make sure 'reindex' is not called for no reason, since the nodes IDs will fill a complete range
        args.reindex = False
        # if use_weights is enabled, automatically make the network to be created weighted
        args.weighted = args.weighted or args.use_weights
        # dynamic edge updates in this setting gets enabled only when the args.edge_sample_size list has entries
        args.is_dynamic &= bool(args.edge_sample_size)

    # the following block is for random graphs, which are not dynamic and for which we can use `args.netsize` to sample the first infected
    # OR for predefined networks that have the `nid` supplied as part of `nettype`
    # note that an `infseed` also needs to be supplied; if not, the first infected will be sampled at network creation time
    # being sampled at net creation time, the first infected become dependent on the `args.netseed` + `network_index`
    # and therefore they can be DIFFERENT for each network (useful for averaging purposes)
    if is_nids_known:
        # update number of first infected to reflect absolute values rather than percentage
        # if args.first_inf >= 1 just turn the value into an int and use that as number of first infected
        args.first_inf = int(args.first_inf if args.first_inf >= 1 else args.first_inf * args.netsize)
        if args.infseed is not None:
            if args.infseed >= 0:
                if args.reindex:
                    nodes = range(args.netsize)
                # Random first infected across simulations - seed random locally
                first_inf_nodes = random.Random(args.infseed).sample(nodes, args.first_inf)
            else:
                first_inf_nodes = []

    # turn off multiprocessing if only one net and one iteration selected
    if args.nnets == 1 and args.niters == 1: args.multip = 0
    # if multiprocessing selected, stop drawing
    if args.multip: 
        args.draw = args.draw_iter = 0
    # otherwise, if animation of the infection progress is selected, disable all prints and enable both draw types
    elif args.animate == 1:
        if not args.draw: args.draw = 1
        # if no draw_iter selected, set the sleep time between iters to 1
        if not args.draw_iter: args.draw_iter = .5

    # if no dual, this means the tracing network effectively has full overlap, while the second tracing network is disabled
    if not args.dual:
        args.overlap = 1.
        args.overlap_two = -1.
 
    if not args.gamma: args.spontan = False
    if not args.gammatau: args.gammatau = args.gamma
    
    # if age-group dependent vars have been provided as array, then choose the value based on inputted age-group 
    if not np.isscalar(args.ph): args.ph = (args.ph[args.group] if args.group is not None else np.average(args.ph, weights=args.group_percent))
    if not np.isscalar(args.lamdahr): args.lamdahr = (args.lamdahr[args.group] if args.group is not None else np.average(args.lamdahr, weights=args.group_percent))
    if not np.isscalar(args.lamdahd): args.lamdahd = (args.lamdahd[args.group] if args.group is not None else np.average(args.lamdahd, weights=args.group_percent))
    
    # we can simulate with a range of tracing rates or with a single one, depending on args.taut supplied
    tracing_rates = np.atleast_1d(args.taut)
    
    # this flag is utilized to determine which type of Pool object to use for multiprocessing (if True, torch Tensors will be shared across processes)
    # moreover, this flag determines whether ranking Model is initialized here (during training, shared across processes), or later on (not shared)
    args.is_learning_agent = False
    # If an Agent was supplied, configure Simulator as such
    if args.agent:
        # allow for modifications over the agent dict without modifying the original reference
        args.agent = args.agent.copy()
        # disable testing and tracing from the random exponential sampling
        args.taur = args.noncomp = 0
        # default dual = 2 to dual = 1 (tracing happens over one network only)
        args.dual = ceil(args.dual / 2)
        # if unset, assign args.taut to the epsilon sequence for each episode as per the control_schedule
        if args.control_schedule and tracing_rates[0] == 0:
            n_episodes, eps_start, eps_decay = args.control_schedule
            args.taut = tracing_rates = [round(eps_start * eps_decay**i, 4) for i in range(int(n_episodes))]
            
        # create tb_layout structure for tensorboard if logging enabled (i.e. ckp_ep or ckp != 0)
        if args.agent.get('ckp_ep', 0) or args.agent.get('ckp', 0):
            args.agent['tb_layout'] = {'Ep infected': OrderedDict(), 'Ep loss': OrderedDict()}
            
        # the type of the agent is needed for unique logic paths
        agent_type = args.agent.get('typ', '')
        
        args.is_learning_agent = any((agent_type.__contains__(typ) for typ in ('sl', 'rl', 'mix'))) and args.agent.get('lr', 0) > 0
        # init learning model if supplied and needed (this ensures Model tensors are shared across all Agent instances)
        ranking_model = args.ranker if args.ranker else args.agent.get('ranking_model', None)
        if args.is_learning_agent and isinstance(ranking_model, dict):
            from .tracing.agent_factory import create_model
            if 'source' not in ranking_model:
                ranking_model['source'] = args.agent.get('source', 'control_diffusion')
            args.agent['ranking_model'] = create_model(k_hops=args.agent.get('k_hops', 2), 
                                                       static_measures=args.agent.get('static_measures', ('degree',)), 
                                                       **ranking_model)
            
        # initialize args.shared if it has not been supplied and it is needed for RecordAgent
        if agent_type.__contains__('rec'):
            args.is_record_agent = True
            if args.shared is None:
                args.shared = Manager().dict()
        # if no record agent, a new shared Manager process is not needed (but we keep it if it's been supplied as param)
        else:
            args.is_record_agent = False
               
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
    
    # unique random tracing rate, relative random tracing factor, and tracing-specific recovery rate (used by different transition functions)
    taur = args.taur
    rel_taur = args.rel_taur
    gammatau = args.gammatau
    # if no taut_two set, delay_two will be used together with all the given taut rates to compute the taut_two rates 
    set_taut_two = (args.taut_two is None)

    for idx, taut in enumerate(tracing_rates):
        # tr_rate=taut will be used as the net.count_importance, which can be used for multiple purposes
        # the id of the current taut will be used by the agent to establish the episode number
        if args.agent:
            args.agent['episode'] = idx
            args.agent['epsilon'] = taut
        elif not args.animate:
            print('\n=========================================================')
            print('Running simulation with parameters:')
            if args.netsize == -1:
                print('WARNING: A predefined network was selected, but node ids were NOT supplied')
            ut.pvar(args.netsize, args.k, args.dual, args.model, owners=False, nan_value=-1)
            ut.pvar(args.overlap, args.uptake, args.maintain_overlap, owners=False, nan_value=-1)
            if args.dual == 2:
                ut.pvar(args.overlap_two, args.uptake_two, args.maintain_overlap_two, owners=False, nan_value=-1)
            ut.pvar(taut, taur, args.noncomp, args.noncomp_dependtime, owners=False, nan_value=-1)
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
                       'expFactorTimesCountImportance', exp=exp, state='T', base=taur*rel_taur)

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
                add_trans(trans_know_items, 'T', 'R', ut.get_stateless_sampling_func(gammatau, exp))
        
        nnets = args.nnets
        net_range = range(nnets)
        # Multiprocessing object to use for each network initialization
        engine = EngineNet(args=args, first_inf_nodes=first_inf_nodes, no_exposed=no_exposed, is_covid=is_covid,
                           tr_rate=taut, trans_true=trans_true_items, trans_know=trans_know_items)
        
        extra_return = [args.shared]
        # for args.multip == 1 or 3, distribution over networks will be performed (either exclusively or inclusively)
        if args.multip in (1, 3):
            # we use normal Pool for distributing only the networks, but NoDaemonPool if we distribute both networks and iters
            # in the first case we use the full cpu count, in the latter we use a half to allow for iters to be distributed
            daemon, jobs = (True, args.cpus) if args.multip == 1 else (False, args.cpus // 2)
            # different pool types are required for different parameter settings
            pool_type = ut.get_pool(pytorch=args.is_learning_agent, daemon=daemon, set_spawn=args.control_gpu)
            if args.agent and args.agent.get('half_cpu', True): jobs //= 2

            with pool_type(jobs) as pool:
                for inet, (net_events, net_info) in enumerate(ut.tqdm_redirect(pool.imap_unordered(engine, net_range),
                                                                               total=nnets,
                                                                               desc='Networks simulation progress')):
                    # Record sim results
                    stats.sim_summary[inet] = net_events
                    # update args with the last network info (making it a proxy for all for logging purposes)
                    if inet == nnets - 1:
                        args.netsize, args.avg_deg, args.overlap, args.overlap_two = net_info
                        # if `k` is still not set to an eligible value, we can safely set it to `args.avg_deg`
                        if args.k <= 0:
                            args.k = args.avg_deg
        else:
            for inet in net_range:
                if nnets > 1:
                    print(f'----- Episode {idx}, simulating network no. {inet} -----')

                # Run simulation, and since no distribution required, return the last networks for inspection purposes
                # extra_return will be a tuple of true_net, know_net
                net_events, true_and_know_nets = engine(inet, return_last_net=True)

                # Record sim results
                stats.sim_summary[inet] = net_events
            # update extra_return with the last networks for inspection purposes
            extra_return.extend(true_and_know_nets)
   
        stats.results_for_param(taut)

    if args.summary_print is None:
        return None
        
    # if summary_splits is set, then we need to compute the full summary statistics for the epidemic
    if args.summary_splits:
        # summary_print will specify whether to print the summary statistics to stdout, a file, or not at all
        statistics = stats.full_summary(args.summary_splits, args.summary_print, args.r_window)
        # we return the statistics and the extra_return tuple (for an API call) if summary_print is configured NOT to print statistics
        return (statistics, extra_return) if args.summary_print == 0 else None
    # otherwise, return the stats object and the extra_return tuple to an API call
    else:
        return stats, extra_return


def parse_args():
    """
    Parse the command line arguments and return the parsed arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    argparser = argparse.ArgumentParser()

    for k in PARAMETER_DEFAULTS.keys():
        default = PARAMETER_DEFAULTS[k]
        if k in {'taut', 'control_schedule', 'edge_sample_size', 'voc_change'}:
            argparser.add_argument('--' + k, type=float, nargs="+", default=default)
        elif k == 'agent':
            argparser.add_argument('--agent', '-a', type=ut.get_json, default=default)
        elif k == 'ranker':
            # special argument that allows for a config file to be used for the ranking learning model
            argparser.add_argument('--ranker', '-r', type=ut.get_json, default=default)
        else:
            # The following is needed since weirdly bool('False') = True in Python
            typed = type(default) if not isinstance(default, bool) else lambda arg: arg.lower() in ('yes', 'true', 't', '1')
            argparser.add_argument('--' + k, type=typed, default=default)
    args, _ = argparser.parse_known_args()

    return args

            
def run_api(**kwargs):
    """
    Run simulations using coding interface.
    
    Args:
        **kwargs: keyword arguments to override default parameters
    
    Returns:
        The result of running the main function with the specified arguments
    """
    argns = argparse.Namespace()

    for k in PARAMETER_DEFAULTS.keys():
        vars(argns)[k] = kwargs.get(k, PARAMETER_DEFAULTS[k])
    
    return main(argns)
    
    
if __name__ == '__main__':

    main()