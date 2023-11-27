import os
import networkx as nx
import inspect
import itertools
import random
from sys import stderr

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex

from .utils import is_not_empty, rand_pairs, rand_pairs_excluding, get_z_for_overlap, get_overlap_for_z, float_defaultdict
from .simulation import Simulation


# in the state names we distinguish between I (SIR/SEIR) and Ip (Covid) only for drawing legend purposes
STATES_NAMES = {
    'S': 'Susceptible',
    'E': 'Exposed',
    'I': 'Infectious',
    'Ip': 'Infectious (presym)',
    'Ia': 'Infectious (asym)',
    'Is': 'Infectious (sym)',
    'H': 'Hospitalized',
    'T': 'Traced',
    'R': 'Recovered',
    'D': 'Dead',
    'N': 'Non-isolating'
}

STATES_COLOR_MAP = {
    'S': '#00F575',
    'E': '#ffcc33',
    'I': '#ff4500',
    'Ip': '#ff4500',
    'Ia': 'pink',
    'Is': 'red',
    'H': 'cyan',
    'T': '#0247fe',
    'R': '#2AAA8A',
    'D': 'gray',
    'N': 'purple'
}

MODEL_TO_STATES = {
    'sir': ['S', 'I', 'R', 'T', 'N'],
    'seir': ['S', 'E', 'I', 'R', 'T', 'N'],
    'covid': ['S', 'E', 'Ip', 'Ia', 'Is', 'H', 'D', 'R', 'T', 'N']
}

ACTIVE_EDGE_COLOR = 'red'
ACTIVE_EDGE_WIDTH = 1.5

    
class Network(nx.Graph):
    """
    A class representing a network of nodes and edges that extends nx.Graph.

    Attributes:
        UNINF_STATES (set): The states that are considered as uninfected, with the default value of {'S'}.
    """

    UNINF_STATES = {'S'}

    def get_simulator(self, *args, **kwargs):
        """
        Returns a Simulation object for the current network.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A Simulation object.
        """
        return Simulation(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        Initializes a Network object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # initialize the network id to 0
        self.inet = 0
        # for sampling dynamic edges
        self.edge_sample_seed = None
        self.edge_sample_size = None
        self.edge_prob_sampling = False
        # whether edges are weighted
        self.weighted = False
        # set use_weights default - whether or not weights of edges are considered in counts
        self.use_weights = False
        # if edges will be dynamic
        self.is_dynamic = False
        # utilized by random networks to keep track of the whole static graph when converting to temporal
        self.links_to_create = None
        # edges added at first timestep
        self.first_edges = None
        # Atomic Counter for node ids
        self.cont = itertools.count()
        # Overlap after noising - initially perfect overlap, MAY change after calling 'noising_links'
        self.overlap = 1
        # Maintain 'active' (not orphans) node list + neighbor counts; and inf + traced states for all nodes
        self.node_list = []
        self.node_states = []
        self.node_infected = []
        self.node_traced = []
        self.node_mask = None
        self.node_sah = None
        self.node_names = None
        # nids -> State -> neighbor counts in State
        self.node_counts = defaultdict(float_defaultdict)
        self.inf_time = {}
        self.traced_time = {}
        self.noncomp_time = {}

        ### Vars that are not copied over by the copy method
        # relative counts importance (will be multiplied with the counts to obtain their importance in the rate)
        self.count_importance = 1
        # Normalization term used in weight normalization
        self.norm_factor = 1
        # whether this is an original or a dual network - needed for drawing
        self.is_dual = False
        # contact reduction rate
        self.contact_reduction = 0
        # control_day and control_iter, set by the agent - needed for drawing
        self.control_iter = self.control_day = -1
        # tested and traced nodes, set by the agent - needed for drawing
        self.tested = self.traced = None
        # node rankings - needed for drawing
        self.computed_measures = None
        # colorbar of node scores - needed for drawing
        self.cbar = None
        # Initialize pos to None - a new spring layout is created when first drawn
        self.pos = None
        self.pyvis_graph = None
        super().__init__(*args, **kwargs)
        
    def copy(self):
        """
        Returns a deep copy of the network, including any extra state information.

        Returns:
            A deep copy of the network.
        
        Notes:
        This method relies on the basic copy mechanism of networkx. It uses copy_state_from to copy over extra state information from the true net to the new network.
        """
        # relying on the basic copying mechanism of networkx
        obj = super().copy()
        # use copy_state_from to copy over extra state information from the true net to the new network
        obj.copy_state_from(self)
        return obj

    def subgraph(self, *args, **kwargs):
        """
        Returns a subgraph of the network.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        A subgraph of the network.

        Notes:
        This method relies on the basic subgraph mechanism of networkx. It uses copy_state_from to copy over extra state information from the true net to the new network.
        """
        obj = super().subgraph(*args, **kwargs)
        obj.copy_state_from(self)
        return obj
    
    def copy_state_from(self, obj):
        """
        Copies the important state information of the given network object to the parameter network object.

        Args:
            obj (Network): The network object to copy the state from.

        Returns:
            None
        """
        # instance primitives
        self.inet = obj.inet
        self.weighted = obj.weighted
        self.use_weights = obj.use_weights
        self.is_dynamic = obj.is_dynamic
        self.norm_factor = obj.norm_factor
        self.edge_prob_sampling = obj.edge_prob_sampling
        self.edge_sample_seed = obj.edge_sample_seed
        # lists or dicts which will be SHARED between true net and dual net
        self.cont = obj.cont
        self.node_list = obj.node_list
        self.node_states = obj.node_states
        self.node_infected = obj.node_infected
        self.node_traced = obj.node_traced
        self.node_names = obj.node_names
        self.inf_time = obj.inf_time
        self.traced_time = obj.traced_time
        self.noncomp_time = obj.noncomp_time
        self.links_to_create = obj.links_to_create
        self.edge_sample_size = obj.edge_sample_size
        # node_counts is the only reference not copied over from the original network
        self.node_counts = defaultdict(float_defaultdict)
        
    def init_random(self, n=200, k=10, typ='random', p=.1, weighted=False, seed=None, edge_sample_size=None, edge_prob_sampling=True):
        """
        Initializes a random network of n nodes and k average degree based on the type of network specified.

        Args:
            n (int): Number of nodes in the network.
            k (int): Average degree of the network.
            typ (str): Type of network to create. Supported types are:
                - 'random': Random network.
                - 'binomial': Binomial network.
                - 'ws': Small-world network.
                - 'newman-ws': Newman-Watts-Strogatz network.
                - 'barabasi': Barabasi-Albert network.
                - 'barabasi:{m_1}:{m_2}': Dual-Barabasi-Albert network with `m_1` and `m_2` supplied in the string.
                - 'powerlaw-cluster': Powerlaw cluster network.
                - 'complete': Fully connected network.
                - 'sbm:{c}:{intra_p}:{partition_inf}': Stochastic block model network. `c` represents the number of communities, 
                    `intra_p` is the intracommunity edge probability, and `partition_inf` is the partition index to use. 
                    Argument `p` dictates the unique intercommunity edge probability.
                - 'config:{weights}:{norm-settings}:{powerlaw-settings}': Configuration model network. `weights` is a comma-separated 
                    pair of weights for the normal and powerlaw dist, which are sampled to obtain the degree sequence, 
                    `norm-settings` is a comma-separated pair of mean and std for the normal dist, and `powerlaw-settings` is a powerlaw exponent.
            p (float): Probability of (re)wiring edges in the network (how this is used depends on `typ`).
            weighted (Union[bool,list]): If True, the edge weights are sampled from uniform(.5, 1). If False, the edges are unweighted.
                If a list is supplied, the edge weights are sampled from a mixture of norm(*weighted[0]) and beta(*weighted[1])
            seed (int): Seed for the random number generator.
            edge_sample_size (list or tuple): A sequence of either 1 or 2 elements. If 1 element, it designates the edge sample size/percentage per increment. 
                If 2, they are interpreted as the a, b limits of a uniform distribution U[a,b], which upon sampling renders the edge sample size/percentage.
            edge_prob_sampling (bool): Whether to sample edges probabilistically, where edges are selected with probability `edge_sample_size`,
                or deterministically, where a fraction of `edge_sample_size` edges from the total edge list are selected at random.

        Returns:
            None
        """
        # add n nodes to the network in the start state -> S
        self.add_mult(n, state='S', traced=False)
        # initialize the rewire probability to k/n if p=None provided
        p = k / n if p is None else p
        
        if 'sbm' in typ:
            splits = typ.split(':')
            try:
                # number of communities should be specified after ':'
                n_community = int(splits[1])
            except (IndexError, ValueError):
                raise ValueError('When an SBM network is selected, one needs to specify after ":" the number of communities')
            try:
                intra_prob = float(splits[2])
            except (IndexError, ValueError):
                # default is 'k'/'n' as the intracommunity edge probability
                intra_prob = k / n
            try:
                partition_inf = int(splits[3])
            except (IndexError, ValueError):
                partition_inf = 1
            typ = 'sbm'
            # divide n into 'n_community' number of communities (without losing any node due to inexact division)
            div, mod = divmod(n, n_community)
            sizes = [div + (1 if i < mod else 0) for i in range(n_community)]
            # probabilities in SBMs come in the form of a matrix relating communities to each other
            # we utilize 'p' as the intercommunity edge probability
            probs = np.ones((n_community, n_community)) * p
            np.fill_diagonal(probs, intra_prob)
            # we create the graph here such that we can remember the partitions
            block_model = nx.stochastic_block_model(sizes, probs, seed=seed)
            if partition_inf:
                self.graph['partition'] = block_model.graph['partition'][partition_inf - 1]
        elif 'barabasi:' in typ:
            splits = typ.split(':')
            infer_p_matching_k = True
            try:
                m1 = int(splits[1])
            except (IndexError, ValueError):
                m1 = int(k)
                infer_p_matching_k = False
            try:
                m2 = int(splits[2])
            except (IndexError, ValueError):
                m2 = int(k)
                infer_p_matching_k = False
            typ = 'dual-barabasi'
            if infer_p_matching_k:
                # based on formula in Dual-Barabasi paper
                p = round((k / (2 * (n - max((m1,m2)))) * n - m2) / (m1 - m2), 6)
        elif 'config' in typ:
            # typ should have a format like 'config:{weights}:{norm-settings}:{powerlaw-settings}'
            splits = typ.split(':')
            try:
                norm_w, power_w = map(float, splits[1].split(','))
            except (IndexError, ValueError):
                norm_w, power_w = (p, 1-p)
            try:
                norm_mean, norm_std = map(float, splits[2].split(','))
            except (IndexError, ValueError):
                norm_mean, norm_std = (0, 1)
            dist = ss.norm(norm_mean, norm_std)
            try:
                power_a = float(splits[3])
            except (IndexError, ValueError):
                power_a = .5
            typ = 'config'
            # the possible degree range will be from 1 to max_degree (controlled via k, but cannot be larger than n - 1)
            deg_range = range(1, min(int(k) + 1, n))
            print(norm_w, power_w, power_a, max(deg_range))
            p = np.fromiter((norm_w * dist.pdf(i) + power_w * i ** (-power_a) for i in deg_range), dtype=float)
            p /= p.sum()
            deg_sequence = np.random.RandomState(seed).choice(deg_range, size=n, replace=True, p=p)
            print(sum(deg_sequence))
            if sum(deg_sequence) % 2:
                deg_sequence[0] += 1
                
        links_to_create_dict = {
            # Generate random pairs (edges) without replacement
            'random': lambda: rand_pairs(n, int(n * k / 2), seed=seed),
            'binomial': lambda: list(nx.fast_gnp_random_graph(n, p, seed=seed).edges),
            # small-world network
            'ws': lambda: list(nx.watts_strogatz_graph(n, int(k), p, seed=seed).edges),
            'newman-ws': lambda: list(nx.newman_watts_strogatz_graph(n, int(k), p, seed=seed).edges),
            # scale-free network
            'barabasi': lambda: list(nx.barabasi_albert_graph(n, m=int(k), seed=seed).edges),
            'dual-barabasi': lambda: list(nx.dual_barabasi_albert_graph(n, m1=m1, m2=m2, p=p, seed=seed).edges),
            'powerlaw-cluster': lambda: list(nx.powerlaw_cluster_graph(n, m=int(k), p=p, seed=seed).edges),
            # fully connected network
            'complete': lambda: list(nx.complete_graph(n).edges),
            # stochastic block model - communities
            'sbm': lambda: list(block_model.edges),
            # config model
            'config': lambda: list(configuration_model(deg_sequence, seed=seed).edges)
        }
        try:
            self.links_to_create = links_to_create_dict[typ]()
        except KeyError:
            print("The inputted network type is not supported. Default to: random", file=stderr)
            self.links_to_create = links_to_create_dict['random']()
                    
        # hyperparameters needed to perform edge sampling over time
        self.weighted = weighted
        self.edge_sample_seed = seed
        self.edge_sample_size = edge_sample_size
        self.edge_prob_sampling = edge_prob_sampling
        # perform sampling based on self.links_to_create and the above hyperparameters
        self.add_links(self.sample_edges(update_iter=0), update=False)
        # record first set of edges
        self.first_edges = list(self.edges(data=True))
        
    def sample_edges(self, update_iter=0):
        """
        Samples a subset of edges from the list of links to create, and optionally assigns random weights to the edges.

        Args:
            update_iter (int): An integer value used to seed the random number generator for edge sampling, meant to signify different update itertions.

        Returns:
            A list of tuples representing the sampled edges. Each tuple contains two node IDs and an optional weight value.
        """
        weighted = self.weighted
        edge_rng = np.random.RandomState(self.edge_sample_seed + update_iter if self.edge_sample_seed is not None else None)
        edge_sample_size = self.edge_sample_size
        if edge_sample_size and edge_sample_size[0] > 0:
            if len(edge_sample_size) == 1:
                edge_sample_size = edge_sample_size[0]
            else:
                edge_sample_size = edge_sample_size[0] + (edge_sample_size[1] - edge_sample_size[0]) * edge_rng.random()
            len_links = len(self.links_to_create)
            if self.edge_prob_sampling:
                indices = np.nonzero(edge_rng.random(len_links) <= edge_sample_size)[0]
            else:
                indices = edge_rng.choice(len_links, int(edge_sample_size if edge_sample_size >= 1 else edge_sample_size * len_links), replace=False)
            edges = [self.links_to_create[i] for i in indices]
        else:
            edges = self.links_to_create
        # Add the random edges with/without weights depending on 'weighted' parameter
        if weighted:
            # default of the next operation is when weighted = [(), ()]
            if isinstance(weighted, list) or isinstance(weighted, tuple):
                norm_w, norm_l, norm_a = weighted[0] if weighted[0] else (.47, .41, .036)
                ndist = ss.norm(norm_l, norm_a)
                beta_w, beta_l, beta_a = weighted[1] if len(weighted) > 1 and weighted[1] else (.53, 5.05, 20.02)
                bdist = ss.beta(beta_l, beta_a)
                w_sum = norm_w + beta_w
                norm_w /= w_sum
                beta_w /= w_sum
                # A stream of indices from which to choose the component
                mixture_idx = edge_rng.choice(2, size=len(edges), replace=True, p=(norm_w, beta_w))
                # y is the mixture sample
                weights = np.fromiter((ndist.rvs(random_state=edge_rng) if i == 0 else bdist.rvs(random_state=edge_rng) for i in mixture_idx), dtype=np.float32)
            else:
                # Create random weights from .5 to 1 and append them to the list of edge tuples
                weights = edge_rng.uniform(.5, 1, size=len(edges)).astype(np.float32)
            # Append weights to the list of edge tuples
            edges = [e + (weights[i],) for i, e in enumerate(edges)]
        return edges

    def noising_links(self, overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, conserve_overlap=True, update=True, seed=None, active_based_noising=False):
        """
        Adds or removes edges from the network in a noisy manner, based on the given parameters.

        Args:
            overlap (float): The desired overlap between a tracing subview and the true network. If not provided, it will be calculated based on the average degree 
                and the z_add and z_rem parameters.
            z_add (float): The expected number of edges to add per node. Default is 0.
            z_rem (float): The expected number of edges to remove per node. Default is 5.
            keep_nodes_percent (float): The percentage of nodes whose edges should be kept in the network, used to model the digital uptake. Default is 1 (100%).
            conserve_overlap (bool): Whether to try to maintain the overlap value while removing edges through `keep_nodes_percent`. Default is True.
            update (bool): Whether to update the neighborhood counts of infections after adding or removing edges. Default is True.
            seed (int): The seed for the random number generator. Default is None.
            active_based_noising (bool): Whether to use only active non-orphan nodes for noising links operations. Default is False.

        Returns:
            None
        """
        # recover the number of nodes for which noising links operations will be performed
        # note that 'active_based_noising' parameter can be used to change between using self.nodes (all nodes) or self.node_list (only active nonorphan nodes)
        nodes = self.node_list if active_based_noising else self.nodes
        n = len(nodes)
        # recover average degree
        k = self.avg_degree()
        
        # current edge set (prior to noising, nodes in each edge tuple are sorted)
        current_edges = {tuple(sorted(edge)) for edge in self.edges}
        
        # links to add and rem lists
        links_to_add, links_to_rem = [], []
        remaining_links_rem = 0
        
        # seed local random if available
        rand = random if seed is None else random.Random(seed)
        
        # if we are interested in conserving the overlap, then the correct z_add and z_rem will be calculated based on the params
        if conserve_overlap:
            # if no overlap value, use z_add & z_rem
            if overlap is None or overlap == -1:
                # average degree k, z_add, z_rem used to calculate overlap after noising
                self.overlap = get_overlap_for_z(k, z_add, z_rem)
            # If overlap given, calculate z_add and z_rem; SUPPLIED z_add and z_rem are mostly IGNORED
            else:
                self.overlap = overlap
                # if z_add = 0, z_add will stay 0 while z_rem = z; OTHERWISE z_add = z_rem = z
                z_add, z_rem = get_z_for_overlap(k, overlap, include_add=z_add)
                
            # random adding z_add on average - this should be 0 in case of digital tracing networks
            remaining_links_add = int(n * z_add / 2)
            links_to_add = rand_pairs_excluding(n, remaining_links_add, to_exclude=current_edges, seed=seed) if z_add else []

            # random removing z_rem on average - this will be preserved even if keep_nodes_percent < 1
            remaining_links_rem = int(n * z_rem / 2)
            #
            # the logic for removing edges is done below (need to account for keep_nodes_percent value)
        else:
            self.overlap = 1.
        
        # simple case, no node's edges are removed completely from the network unless the overlap does it
        if keep_nodes_percent == 1:
            links_to_rem = rand.sample(current_edges, remaining_links_rem)
        # the links of certain nodes will be completely removed if keep_nodes_percent is below 1 (<100%)
        else:
            untraceable_nodes = rand.sample(nodes, int(round(1 - keep_nodes_percent, 2) * n))
            untraceable_edges = {tuple(sorted(edge)) for edge in self.edges(untraceable_nodes)}
            len_untraceable_edges = len(untraceable_edges)
            
            # update the total links_to_rem list
            links_to_rem += untraceable_edges
            
            # conserve_overlap tries to ensure the overlap value is also met in the keep_nodes_percent < 1 case
            # if this value cannot be met, an error will be thrown
            if conserve_overlap:
                remaining_links_rem -= len_untraceable_edges
                if remaining_links_rem >= 0:
                    # The rest of the edges until the overlap value is met are to be removed at random
                    links_to_rem += rand.sample(current_edges - untraceable_edges, remaining_links_rem)
                else:
                    raise ValueError("The value of keep_nodes_percent is too small for maintaining the selected overlap!")
            # in the case the supplied overlap gets ultimately ignored, we calculate the true value for it (for information purposes)
            else:
                actual_z_add = 0
                actual_z_rem = len_untraceable_edges * 2 / n
                self.overlap = get_overlap_for_z(k, actual_z_add, actual_z_rem)
                
        self.add_edges_from(links_to_add)
        self.remove_edges_from(links_to_rem)

        # initially there is no one traced that we care about, so we only update counts for infection status
        if update:
            self.update_counts()
    
    def init_for_simulation(self, first_inf_nodes, dynamic=False):
        """
        Initializes the network for a simulation, setting all nodes to the 'S' (Susceptible) state and tracing status to False.
        If `dynamic` is True, clears all current edges and adds back the first set of edges.
        If `first_inf_nodes` is not empty, changes the state of those nodes to 'I' (Infected) and updates the infection status counts.

        Args:
            first_inf_nodes (list): A list of node IDs to be initially infected.
            dynamic (bool, optional): Whether the graph is dynamic, where one would need to readd the first set of edges back. Defaults to False.
        """
        # this will initialize states to 'S' and tracing status to False
        self.init_states(state='S')
        if dynamic:
            self.clear_edges()
            # add back the first set of edges
            self.add_edges_from(self.first_edges)
        if first_inf_nodes:
            # initially, there should be no one traced so we only update the infection status counts
            self.change_state(first_inf_nodes, state='I', update=True)
    
    def init_states(self, state='S'):
        """
        Initializes the state of all nodes in the network to the given state, reinitializes the tracing states according to the current list of non-orphans,
            and clears the infection, tracing, and noncompliance times.

        Parameters:
        state (str): The state to which all nodes in the network should be initialized. Default is 'S'.

        Returns:
        None
        """
        len_to_max_id = max(self) + 1
        list_to_max_id = range(len_to_max_id)
        self.node_states = [state] * len_to_max_id
        self.node_infected = [state not in self.UNINF_STATES] * len_to_max_id
        # the tracing states array must reflect that only the active nodes are traceble
        self.node_traced = np.isin(list_to_max_id, self.node_list, invert=True).tolist()
        self.inf_time = {}
        self.traced_time = {}
        self.noncomp_time = {}

    def add(self, state='S', traced=False, ids=None, update=False):
        """
        Adds a node to the network with the given state and traced status.

        Args:
            state (str, optional): The state of the node to be added. The default is 'S', which means susceptible.
            traced (bool, optional): The traced status of the node to be added. The default is False, which means not traced.
            ids (int, optional): The id(s) of the node(s) to be added. The default is None, which means a new id will be generated from an atomic counter.
            update (bool, optional): A flag that indicates whether to update the neighborhood counts after adding the node. The default is False, which means no update.

        Returns:
            None
        
        Notes:
            This method will attempt to add a node no matter what is already in the network.
        """
        if ids is None:
            # do-while loop equivalent - makes sure a node is added no matter what is already in the network
            while True:
                ids = next(self.cont)
                if ids not in self.nodes:
                    break
        # if ids was supplied, we just add them to the network as-is
        # if no ids supplied, a fitting ids was generated by this point
        self.add_node(ids)

        # make sure the node_states and node_traced lists cover this added position by filling the lists with defaults
        needed_len_state_list = ids + 1 - len(self.node_states)
        if needed_len_state_list > 0:
            self.node_states += ['S'] * needed_len_state_list
            self.node_infected += [False] * needed_len_state_list
            self.node_traced += [False] * needed_len_state_list
        # finally amend the 'state' and 'traced' status of this added node in the lists
        self.node_states[ids] = state
        self.node_infected[ids] = state not in self.UNINF_STATES
        self.node_traced[ids] = traced
        
    def add_mult(self, n=200, state='S', traced=False, ids=None, update_existing=False):
        """
        Adds multiple nodes to the network.

        Args:
            n (int, optional): The number of nodes to add if `ids` is not provided. Default is 200.
            state (str, optional): The state to assign to the new nodes. Default is 'S'.
            traced (bool, optional): Whether the new nodes should be marked as traced. Default is False.
            ids (list, optional): A list of node IDs to add. If provided, `n` is ignored.
            update_existing (bool, optional): Whether to update the state and traced status of existing nodes based on this operation. Default is False.

        Returns:
            None
        
        Notes:
            If list of `ids` is not provided, this method guarantees that `n` additions will be performed past the max ID or current counter.
            If list of `ids` is supplied, it will be given priority, so `n` will be effectively ignored.
        """
        if ids is None:
            # first element to be added is either max(nid) + 1 OR current_count + 1 = next_count - 1 + 1
            nxt = max(self.nodes, default=next(self.cont)-1) + 1
            ids = range(nxt, nxt + n)
            # increment count to reflect n additions
            self.cont = itertools.count(nxt + n)
        self.add_nodes_from(ids)
        # the following makes sure node_states and node_traced lists are covered for the new additions
        needed_len_state_list = max(ids) + 1 - len(self.node_states)
        if needed_len_state_list > 0:
            self.node_states += ['S'] * needed_len_state_list
            self.node_infected += [False] * needed_len_state_list
            self.node_traced += [False] * needed_len_state_list
        # populate lists with correct state and traced status for ids in question
        is_inf = state not in self.UNINF_STATES 
        for i in ids:
            # update_existing = True -> all nodes get 'state' and 'traced' status from the parameters
            # otherwise: check if node was NOT in the set of active nodes prior to this update (note self.node_list not yet updated here)
            if update_existing or i not in self.node_list:
                self.node_states[i] = state
                self.node_infected[i] = is_inf
                self.node_traced[i] = traced

    def update_counts(self, nlist=None):
        """
        Update the neighborhood state counts of nodes in the network.

        Args:
            nlist (list): List of node IDs to update counts for. If None, update counts for all active nodes.

        Returns:
            None
        """
        # local for efficiency
        # revert node_counts to default
        counts = self.node_counts = defaultdict(float_defaultdict)
        states = self.node_states
        # if nlist not given, update_counts for all 'active' nodes and update W_all
        if nlist is None:
            nlist = self.node_list
        
        for nid in nlist:
            # neighboring counts, will retain a weighted average for each infectious state based on edges weight
            counts_nid = counts[nid]
            for _, neigh, data in self.edges(nid, data=True):
                counts_nid[states[neigh]] += data['weight'] if self.use_weights else 1
            
    def update_counts_with_traced(self, nlist=None):
        """
        Update the neighborhood state counts of nodes in the network, taking into account the separated traced status.

        Args:
            nlist (list): List of node IDs to update counts for. If None, update counts for all active nodes.

        Returns:
            None
        """
        # local for efficiency
        # revert node_counts to default
        counts = self.node_counts = defaultdict(float_defaultdict)
        states = self.node_states
        # if nlist not given, update_counts for all 'active' nodes
        if nlist is None:
            nlist = self.node_list
            
        for nid in nlist:
            # neighboring counts, will retain a weighted average for each infectious state based on edges weight
            counts_nid = counts[nid]
            for _, neigh, data in self.edges(nid, data=True):
                weight = data['weight'] if self.use_weights else 1
                # if the neighboring node is traced and isolated, we do not update the infectious counts of nid
                if not self.node_traced[neigh]:
                    counts_nid[states[neigh]] += weight
                # if we still track the traced time of neighbor, then it still counts for the tracing progression
                if neigh in self.traced_time:
                    counts_nid['T'] += weight
            
    def get_count(self, nid, state='I'):
        """
        Returns the count of neighbors in a given state of the supplied node ID.

        Args:
            nid (int): The ID of the node to get the neighborhood counts for.
            state (str, optional): The state of the neighbors to get the count for. Defaults to 'I'.

        Returns:
            int: The count of nodes with the given ID and state.
        """
        return self.node_counts[nid][state]
        
    def get_state(self, nid):
        """
        Returns the state of the node with the given ID.

        Args:
            nid (int): The ID of the node.

        Returns:
            str: The state of the node with the given ID.
        """
        return self.node_states[nid]
        
    def change_state(self, nids, state='I', update=True, update_with_traced=False, time=0):
        """
        Changes the state of the specified nodes to the given state.

        Args:
            nids (int or array-like of int): The node ID(s) to change the state of.
            state (str, optional): The state to change the node(s) to. Defaults to 'I'.
            update (bool, optional): Whether to update the neighborhood counts of nodes in each state. Defaults to True.
            update_with_traced (bool, optional): Whether to update the neighborhood counts of nodes in each state, acknowledging the separated traced status. 
                Defaults to False.
            time (int, optional): The time at which the node(s) entered the 'E' state. Defaults to 0.
        """
        is_inf = state not in self.UNINF_STATES
        for nid in np.atleast_1d(nids):
            self.node_states[nid] = state
            self.node_infected[nid] = is_inf
            if state == 'E':
                self.inf_time[nid] = time
            elif is_inf and time == 0:
                self.inf_time[nid] = 0
        if update:
            self.update_counts_with_traced() if update_with_traced else self.update_counts()
                
    def change_state_fast_update(self, nid, state, time=0):
        """
        (Fast) update the state of a node in the network and its neighbors' counts, if applicable.

        Args:
            nid (int): The ID of the node to update.
            state (str): The new state of the node.
            time (int, optional): The time at which the node entered the Exposed state, if applicable. Defaults to 0.

        Returns:
            None

        Notes:
            This does not update the counts of neighbors in states that do not impact the transmission. As such, this is a faster alternative to `change_state`,
            but it may lead to inconsistencies in the counts of neighbors in states that do not impact the transmission.
        """
        # local vars for efficiency
        states = self.node_states
        counts = self.node_counts
        # remember old state
        old_state = states[nid]
        
        if state == 'E':
            self.inf_time[nid] = time
        # update neighbor counts if NOT moving to Exposed (We care only about INFECTIOUS transitions)
        # update neighbor counts only if NOT traced / NOT hospitalized CURRENTLY
        elif not self.node_traced[nid] and old_state != 'H':
            for _, neigh, data in self.edges(nid, data=True):
                # update counts only for Suscpetibles or if the state transitions correspond to a tracing event
                if states[neigh] == 'S' or state in ('T', 'N'):
                    # get the counts dict of the neighbor
                    neigh_counts = counts[neigh]
                    # count_val becomes the normalized weight of this edge if self.use_weight
                    count_val = data['weight'] if self.use_weights else 1
                    neigh_counts[old_state] -= count_val
                    # because of fp operations, values may fall within a few decimal places away from 0, when in fact the value is supposed to be 0
                    if abs(neigh_counts[old_state]) < 1e-6:
                        neigh_counts[old_state] = 0.
                    neigh_counts[state] += count_val
                    
        # update the actual node state
        states[nid] = state
        self.node_infected[nid] = state not in self.UNINF_STATES
    
    def change_traced_state_update_tracing(self, nid, to_traced, time_of_trace=None, legal_isolation_exit=False):
        """
        Changes the traced status of a node and updates the tracing counts for the node's neighbors within a TRACING network.

        Args:
            nid (int): The ID of the node to change the traced status of.
            to_traced (bool): The new traced status of the node.
            time_of_trace (float, optional): The time of the trace event. Defaults to None.
            legal_isolation_exit (bool, optional): Whether the node is legally exiting isolation. Defaults to False.

        Returns:
            None

        Notes:
            When using separate_traced, the T status is independent of the other transmission states, represented here via a flag.
            Counts are updated as-if this would be an actual infection state change: 
            i.e. for current infection state of nid decrease/increase count, for 'T' increment/decremenet based on to_traced True/False.
        """
        # switch traced flag for the current node
        self.node_traced[nid] = to_traced

        # If this is a traced event, count_val = 1 | edge_weight, update time of tracing
        if to_traced:
            count_val = 1
            self.traced_time[nid] = time_of_trace
        # for noncomploance, count_val = -1 | -edge_weight
        else:
            count_val = -1
            # we silently remove the node from the tracing time record if the transition to N is due to legal isolation exit
            if legal_isolation_exit:
                self.traced_time.pop(nid, None)
            self.noncomp_time[nid] = time_of_trace

        # Note: we update the counts of traced only if T event OR N event with legal_isolation_exit=True (i.e. when exiting self-isolation)
        # This is to say that becoming N only impacts infection progression not tracing progression (i.e. contacts can still be inferred)
        if to_traced or legal_isolation_exit:
            # local vars for efficiency
            counts = self.node_counts
            for _, neigh, data in self.edges(nid, data=True):
                neigh_counts = counts[neigh]
                # final_count_val becomes the normalized weight of this edge if self.use_weight, with sign dictated by count_val
                neigh_counts['T'] += count_val * data['weight'] if self.use_weights else count_val
                # because of fp operations, values may fall within a few decimal places to 0, when in fact the value is supposed to be 0
                if abs(neigh_counts['T']) < 1e-6:
                    neigh_counts['T'] = 0.
               
    def change_traced_state_update_infectious(self, nid, to_traced, time_of_trace=None, legal_isolation_exit=False):
        """
        Changes the traced status of a node and updates the tracing counts for the node's neighbors within the TRAMSMISSION network.

        Args:
            nid (int): The ID of the node to change the traced status of.
            to_traced (bool): The new traced status of the node.
            time_of_trace (float, optional): The time of the trace event. Defaults to None.
            legal_isolation_exit (bool, optional): Whether the node is legally exiting isolation. Defaults to False.

        Returns:
            None

        Notes:
            When using separate_traced, the T status is independent of the other transmission states, represented here via a flag.
            Counts are updated as-if this would be an actual infection state change: 
            i.e. for current infection state of nid decrease/increase count, for 'T' increment/decremenet based on to_traced True/False.
        """
        # switch traced flag for the current node
        self.node_traced[nid] = to_traced
        inf_state = self.node_states[nid]
        if inf_state.__contains__('I'):
            # If this is a traced event, count_val = 1 | edge_weight, update time of tracing
            count_val = 1 if to_traced else -1
            # local vars for efficiency
            counts = self.node_counts
            # get this node's current infection network state
            for _, neigh, data in self.edges(nid, data=True):
                # final_count_val becomes the normalized weight of this edge if self.use_weight, with sign dictated by count_val
                counts[neigh][inf_state] -= count_val * data['weight'] if self.use_weights else count_val

    def add_link(self, nid1, nid2, weight=1, update=False, update_with_traced=False):
        """
        Add a link between two nodes in the network.

        Args:
            nid1 (int): ID of the first node.
            nid2 (int): ID of the second node.
            weight (float, optional): Weight of the link. Defaults to 1.
            update (bool, optional): Whether to update the neighborhood counts of the nodes. Defaults to False.
            update_with_traced (bool, optional): Whether to update the neighborhood counts of the nodes, acknowledging the traced status. Defaults to False.
        """
        self.add_edge(nid1, nid2, weight=weight)
        if update:
            self.update_counts_with_traced([nid1, nid2]) if update_with_traced else self.update_counts([nid1, nid2])
            
    def rem_link(self, nid1, nid2, update=False, update_with_traced=False):
        """
        Remove a link between two nodes in the network.

        Args:
            nid1 (int): ID of the first node.
            nid2 (int): ID of the second node.
            weight (float, optional): Weight of the link. Defaults to 1.
            update (bool, optional): Whether to update the neighborhood counts of the nodes. Defaults to False.
            update_with_traced (bool, optional): Whether to update the neighborhood counts of the nodes, acknowledging the separated traced status. Defaults to False.
        """
        self.remove_edge(nid1, nid2)
        if update:
            self.update_counts_with_traced([nid1, nid2]) if update_with_traced else self.update_counts([nid1, nid2])
            
    def add_links(self, lst, weight=1., update=False, update_with_traced=False, reindex=False):
        """
        Adds edges to the graph from a list of edges.

        Args:
            lst (list): A list of edges to add to the graph. Each edge can be a tuple of two nodes or a tuple of three elements, 
                where the third element is the weight of the edge.
            weight (float, optional): The weight to assign to the edges if `lst` is a tuple of two nodes. Defaults to 1.0.
            update (bool, optional): Whether to update the neighborhood counts of the graph after adding the edges. Defaults to False.
            update_with_traced (bool, optional): Whether to update the neighborhood counts, acknowledging the separated traced status. Defaults to False.
            reindex (bool, optional): Whether to reindex the nodes in the edges using the `names_to_node` dictionary. Defaults to False.

        Note:
            With `norm_factor` <= 0, normalization of weights is cancelled and the edges will be added to the graph as-given.
        """
        if self.contact_reduction:
            len_lst = len(lst)
            to_remove = len_lst - int(self.contact_reduction if self.contact_reduction >= 1 else self.contact_reduction * len_lst)
            lst = np.array(lst, dtype=object)[np.random.choice(len_lst, to_remove, replace=False)]
        # if edges with no weights are to be added, turn off use_weights flag
        # special logic for reindexing also follows depending on whether weights have been supplied or not
        if len(lst[0]) == 2:
            self.use_weights = False
            if reindex:
                lst = [(self.names_to_node[n1], self.names_to_node[n2]) for n1, n2 in lst]
        elif reindex:
            lst = [(self.names_to_node[n1], self.names_to_node[n2], data) for n1, n2, data in lst]
        # norm_factor used for normalizing weights
        norm_factor = self.norm_factor
        # If the use of weights is turned off or inappropriate norm_factor supplied, add edges in whichever way works given 'lst' and 'weight' without normalization
        if not self.use_weights or norm_factor <= 1:
            try:
                self.add_edges_from(lst, weight=round(weight, 6))
            except TypeError:
                self.add_weighted_edges_from(lst)
        # If we care about the weights, and an appropriate norm_factor exists, normalize weights and add edges accordingly
        else:
            # If a sensible norm_factor is supplied, weighted averaging of the edge weights is carried out
            # We sill allow either the form [1,2,3] or [1,2, {weight:3}]
            try:
                edges_arr = [(n1, n2, round(data['weight'] * norm_factor, 6)) for n1, n2, data in lst]
            except (TypeError, IndexError):
                # the array of edges will hold both ints (nids) and floats (weights)
                edges_arr = np.array(lst, dtype=object)
                edges_arr[:, -1] = (edges_arr[:, -1] * norm_factor).astype(np.float32)
            finally:
                self.add_weighted_edges_from(edges_arr)

        if self.node_mask is not None:
            for n1, n2, data in self.edges(data=True):
                if n1 in self.node_mask:
                    data['weight'] = round(data['weight'] * self.mask_cut, 6)
                if n2 in self.node_mask:
                    data['weight'] = round(data['weight'] * self.mask_cut, 6)
        if self.node_sah is not None: 
            self.remove_edges_from({tuple(sorted(edge)) for edge in self.edges(self.node_sah)})
        
        if update:
            self.update_counts_with_traced() if update_with_traced else self.update_counts()
    
    def infected_neighbors(self, nid=0, filter_traced=True):
        """
        Returns information about the infected neighbors of a given node. This is mainly used for debugging.

        Args:
            nid (int): The ID of the node to check for infected neighbors. Default is 0.
            filter_traced (bool): Whether to exclude traced nodes from the results. Default is True.

        Returns:
            Tuple[int, float, np.ndarray]: A tuple containing the number of infected neighbors, the sum of their weights, and an array of their IDs and weights.
        """
        inf = np.array([[n, self[nid][n]['weight']] for n in self.neighbors(nid) if self.node_states[n].__contains__('I') and not (filter_traced and self.node_traced[n])])
        return (len(inf), inf[:, 1].sum(), inf) if len(inf) else (0, 0, [])
            
    def avg_degree(self, use_weights=False):
        """
        Computes the average degree of the network.

        Args:
            use_weights (bool, optional): If True, the degree is computed using the edge weights. 
                Otherwise, the degree is computed as 2 * number of edges / number of nodes. Default is False.

        Returns:
            float: The average degree of the network.
        """
        if use_weights:
            return np.mean(list(dict(self.degree(weight='weight')).values()))
        else:
            return 2 * len(self.edges) / len(self.nodes)
    
    def compute_efforts(self, taur):
        """
        Computes the accumulated random and active tracing efforts for the network. `taur` is used as a factor for the random tracing effort.
        `self.count_importance` is used as a factor for the active tracing effort.

        Args:
            taur (float): The random tracing effort parameter.

        Returns:
            tuple: A tuple containing the random tracing effort and the active tracing effort.

        Notes:
            The random tracing effort is computed by counting the number of nodes in the 'S', 'E', 'I', 'Ia', 'Is' states that have not been traced yet. 
            The active tracing effort is computed by summing the number of neighbors of nodes in the 'S', 'E', 'I', 'Ia', 'Is' states that have been traced.
        """
        node_states = self.node_states
        node_traced = self.node_traced
        node_counts = self.node_counts
        traceable_states = {'S', 'E', 'I', 'Ia', 'Is'}
        
        # compute random tracing effort -> we only care about the number of traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        rand_effort_accum = 0
        # compute active tracing effort -> we only care about the neighs of nodes in traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        tracing_effort_accum = 0
        for nid in self.node_list:
            current_state = node_states[nid]
            if not node_traced[nid] and current_state in traceable_states:
                rand_effort_accum += 1
                tracing_effort_accum += node_counts[nid]['T']
                
        return round(taur * rand_effort_accum, 2), round(self.count_importance * tracing_effort_accum, 2)
    
    def write_graphml(self, path='fig/graph.graphml', colors=None, **kwargs):
        """
        Writes the network to a GraphML file.

        Args:
            path (str): The path to the file to write to.
            **kwargs: Additional keyword arguments to pass to `nx.write_graphml`.

        Returns:
            None
        """
        if not path.endswith('.graphml'):
            path = f'{os.path.splitext(path)[0]}.graphml'
        for nid in self.nodes:
            self.nodes[nid]['label'] = self.node_states[nid]
            self.nodes[nid]['color'] = to_hex(colors[nid] if colors else STATES_COLOR_MAP[self.node_states[nid]])
        nx.write_graphml(self, path=path, **kwargs)
        
    def draw(self, plotter='default', pos=None, figsize=(12, 10), show=True, ax=None, layout='spring_layout', seed=43, legend=0, with_labels=True, font_size=12, 
             font_color='white', title_fontsize=15, model='covid', node_size=300, degree_size=0, edge_color='gray', edge_width=.7, style='-', bgcolor='moccasin', 
             margins=0, dual_custom=-1, states_color_map={}, layout_kwargs=dict(k=1.8, iterations=150), output_path='fig/graph', **kwargs):
        """
        Draw the Network object using networkx and matplotlib OR plotly OR pyvis.

        Args:
            plotter (str): The method of plotting the network.
                - 'default': display infection and tracing networks separately, 
                - 'custom': display all network state information in one graph, 
                - 'plotly': display all network state information in one using plotly,
                - 'pyvis': display all network state information in one graph using pyvis.
            pos (dict, optiomal): A dictionary with node positions as keys and positions as values. If None, a new layout is generated.
            figsize (tuple): Figure size in inches.
            show (bool): If True, display the plot upon calling this method.
            ax (matplotlib.axes.Axes, optional): A Matplotlib Axes instance to which the plot will be drawn. If None, fig and ax will be created here.
            layout (str): The layout algorithm to use. Default is 'spring_layout'.
            seed (int): Seed for the random number generator.
            legend (int): 0 - no legend; 1 - legend with one letter state name; 2 - legend with full state names.
            with_labels (bool): If True, draw node labels.
            font_size (int): The font size of the node labels.
            font_color (str): The color of the node labels.
            title_fontsize (int): The font size for title elements (e.g. legend).
            model (str): The name of the model.
            node_size (int): The size of the nodes.
            degree_size (int): Factor with which the degree of the nodes influences their displayed size. Default is 0, which means no influence.
            edge_color (str, optional): The color of the edges. If None, default colors of `plotter` method will be used.
            edge_width (float, optional): The width of the edges. If None, default colors of `plotter` method will be used.
            style (str): The style of the edges.
            bgcolor (str): The background color of the plot.
            margins (float): The margins to be used for the `ax` object.
            dual_custom (int): Color configuration for the dual tracing network.
                - <0: full states color
                - 0: only 'I', 'S', and 'T', 'N'
                - 1: node scores as given by externally supplied `computed_measures`, and 'T', 'N'
                - >1: temperature for computing node probs from scores given by `computed_measures, and 'T', 'N'.
            states_color_map (dict): A dictionary with node states as keys and corresponding colors as values.
            output_path (str): The path of the file to save the plot to.
            layout_kwargs (dict): Additional keyword arguments to pass to the layout generator.

        Returns:
            None
        """
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        # get the node state list
        node_states = self.node_states
        states_color_map = {s: states_color_map.get(s, STATES_COLOR_MAP[s]) for s in STATES_COLOR_MAP}
        # for the true network, colors for all nodes are based on their state
        if not self.is_dual:
            colors = [states_color_map[node_states[nid]] for nid in self.nodes]
        # for the dual network, when no explicit T state set, give priority to node_traced == True as if 'T' state
        # also give priority to node_traced == False as if 'N' state IF the node has been traced before (has a traced_time)
        else:
            if dual_custom > 0:
                cmap = plt.cm.YlOrRd
                if self.computed_measures is None:
                    mn = mx = 0
                else:
                    meas = [self.computed_measures.get(nid, 0) for nid in range(len(self))] if isinstance(self.computed_measures, dict) else self.computed_measures
                    mx = max(meas)
                    if dual_custom != 1 and mx > 1:
                        # temperature subtract, max shifting and exponentiate
                        meas = meas / dual_custom
                        meas = np.exp(meas - meas.max())
                        # divide to obtain softmax
                        meas /= meas.sum()
                        mx = max(meas)
                    mn = min(meas)
                norm_color = plt.Normalize(mn, mx)
            colors = []
            for nid in self.nodes:
                if self.node_traced[nid]:
                    color = states_color_map['T']
                elif nid in self.traced_time:
                    color = states_color_map['N']
                elif dual_custom == 0:
                    color = states_color_map['I'] if self.node_infected[nid] else states_color_map['S']
                elif dual_custom > 0:
                    color = cmap(norm_color(meas[nid]) if mx > 0 else 0)
                else:
                    color = states_color_map[node_states[nid]]
                colors.append(color)
        
        if self.is_dynamic and self.links_to_create:
            graph = nx.Graph()
            graph.add_nodes_from(self.nodes)
            # we add all edges in links_to_create to this full static representation, and we assign either 'weight' or 0 to their weight, depending on their presence in `self`
            graph.add_weighted_edges_from([(*e, nx.get_edge_attributes(self, 'weight').get(e, 0.)) for e in self.links_to_create])
            active_edge_set = set(self.edges)
            edge_colors = []
            edge_widths = []
            for edge in self.links_to_create:
                if edge in active_edge_set:
                    edge_colors.append(ACTIVE_EDGE_COLOR)
                    edge_widths.append(ACTIVE_EDGE_WIDTH)
                else:
                    edge_colors.append(edge_color)
                    edge_widths.append(edge_width)
        else:
            graph = self
            edge_colors = edge_color
            edge_widths = edge_width

        # by doing this we avoid overwriting self.pos with pos when we just want a different layout for the drawing
        pos = pos if pos else self.pos
        if pos:
            # if one of self.pos or argument pos was not None, fix the specified node's positions, then generate a new layout around ONLY IF new nodes were added
            if len(self) != len(pos):
                pos = self.generate_layout(graph=graph, pos=pos, fixed=pos.keys(), layout_type=layout, seed=seed, **layout_kwargs)
        else:
            # if both were None, generate a new layout with the seed and use it for drawing
            pos = self.generate_layout(graph=graph, layout_type=layout, seed=seed, **layout_kwargs)
                        
        if degree_size:
            d = nx.degree(self)
            node_size = np.array([(d[node] + 1) * degree_size for node in self.nodes])
        
        legend_config = None
        if legend:
            if legend == 2:
                legend_label = lambda state: STATES_NAMES[state]
                plt.subplots_adjust(left=.2)
            else:
                legend_label = lambda state: state
                plt.subplots_adjust(left=.1)
            legend_config = [(states_color_map[state], legend_label(state)) for state in MODEL_TO_STATES[model]]
            
        # sometimes an iterable of axis may be supplied instead of one Axis object, so only get the first element
        if isinstance(ax, Iterable): ax = ax[0]

        if 'graphml' in plotter:
            self.write_graphml(path=output_path, colors=colors, **kwargs)
        
        if 'pyvis' in plotter:
            # pos_scale can be supplied from the plotter argument and it is used as a scaling factor for the original networkx node positioning
            # if this is not supplied, the default from `draw_pyvis` will be used
            try:
                pos_scale = float(plotter.split(':')[1])
            except (ValueError, IndexError):
                pos_scale = None
            return self.draw_pyvis(graph, output_path=output_path, notebook=True, pos=pos, pos_scale=pos_scale, figsize=figsize, node_color=colors, node_size=node_size/25, 
                                   with_labels=with_labels, font_size=font_size, font_color=font_color, edge_colors=edge_colors, edge_widths=edge_widths, bgcolor=bgcolor, 
                                   legend_config=legend_config, legend_fontsize=title_fontsize)
        elif 'plotly' in plotter:
            # pos_scale can be supplied from the plotter argument and it is used as a scaling factor for the original networkx node positioning
            # if this is not supplied, the default from `draw_plotly` will be used
            try:
                pos_scale = float(plotter.split(':')[1])
            except (ValueError, IndexError):
                pos_scale = None
            return self.draw_plotly(graph, output_path=output_path, notebook=True, pos=pos, pos_scale=pos_scale, figsize=figsize, node_color=colors, node_size=node_size/15,
                                    with_labels=with_labels, font_size=font_size, font_color=font_color, edge_colors=edge_colors, edge_widths=edge_widths, bgcolor=bgcolor, 
                                    legend_config=legend_config, legend_fontsize=title_fontsize)
        else:
            if ax is None and figsize:
                _, ax = plt.subplots(figsize=figsize)
            
            # draw graph
            nx.draw(graph, pos=pos, node_color=colors, node_size=node_size, edge_color=edge_colors, width=edge_widths, style=style,
                ax=ax, with_labels=with_labels, labels=self.node_names, font_size=font_size, font_color=font_color, **kwargs)
            
            if legend:
                empty = mpatches.Rectangle((0, 0), 1, 1, fill=False, linewidth=0, label='')
                ps = [empty]
                ps += [mpatches.Patch(color=config[0], label=config[1]) for config in legend_config]
                ps.append(empty)
                ps.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=.5, label=f'Testing: {self.tested}'))
                ps.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=.5, label=f'Tracing: {self.traced}'))
                # create color legend
                ld = plt.legend(handles=ps, title=f'Control Day: {self.control_iter}', loc='upper left', prop={'size': title_fontsize}, 
                                title_fontsize=title_fontsize, bbox_to_anchor=(0, 1), bbox_transform=plt.gcf().transFigure)
                ld.get_title().set_fontsize(16)

            if dual_custom > 0:
                smap = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap)
                self.cbar = plt.colorbar(smap, fraction=0.07, pad=0, orientation="horizontal", cax=self.cbar.ax if self.cbar else None)

            # ax.margins(margins)
            ax.get_figure().set_facecolor(bgcolor)
            if show:
                plt.show()
                
    def generate_layout(self, graph=None, layout_type='spring_layout', **kwargs):
        """
        Generates a layout for the given graph using the specified layout type and keyword arguments.

        Args:
            graph (networkx.Graph): The graph to generate the layout for. If None, uses the self instance.
                This parameter permits drawing a different graph from self (e.g. a subgraph of active edges if self is dynamic).
            layout_type (str): The type of layout to generate. Can be any of the supported layout types in networkx,
                or a string in the format 'graphviz:prog', where 'prog' is the name of the Graphviz layout program to use.
                If the name does not end with '_layout', '_layout' is appended to it.
            **kwargs: Additional keyword arguments to pass to the layout generator function.

        Returns:
            dict: A dictionary mapping each node in the graph to its (x, y) position in the layout.

        Raises:
            ValueError: If the specified layout algorithm is not recognized by NetworkX.
        """
        if graph is None:
            graph = self
            
        # different logic for graphviz/matplotlib depending on 'layout_type'
        if 'graphviz' in layout_type:
            method_to_call = nx.nx_agraph.pygraphviz_layout
            try:
                kwargs['prog'] = layout_type.split(':')[1]
            except IndexError:
                pass
        else:
            # deals with the case in which the layout was not specified with the correct '_layout' suffix that nx.draw expects
            if not layout_type.endswith('_layout'):
                layout_type += '_layout'
            try:
                # get method from networkx which generates the selected layout
                method_to_call = getattr(nx, layout_type) 
            except AttributeError:
                raise ValueError('This drawing layout is not recognized by networkx.')          
        # get signature of the method to make sure we pass in only the supported kwargs per each layout type
        signature = inspect.signature(method_to_call)
        # skip positional, *args and **kwargs arguments
        skip_kinds = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        # loop through the method signature parameters and either put the value supplied by kwargs or retain the default value
        passed_kwargs = {
            param.name: kwargs[param.name]
            for param in signature.parameters.values()
            if param.name in kwargs and param.kind not in skip_kinds
        }
        
        # generate drawing layout as selected
        self.pos = method_to_call(graph, **passed_kwargs)
        return self.pos
    
    def draw_plotly(self, graph=None, output_path='fig/graph.html', notebook=True, pos=None, pos_scale=1, node_color=None, node_size=10, with_labels=True, font_size=10, 
                    font_color='white', edge_colors=None, edge_widths=None, legend_config=None, legend_fontsize=15, figsize=None, bgcolor='moccasin'):
        """
        Draws a Network object in plotly.

        Args:
            graph (networkx.Graph): The graph to generate the layout for. If None, uses the self instance.
                This parameter permits drawing a different graph from self (e.g. a subgraph of active edges if self is dynamic).
            output_path (str): The path of the output file. Default is 'fig/graph.html'.
            notebook (bool): Whether to display the graph in a Jupyter notebook. Default is True.
            pos (dict): A dictionary of node positions. If None, the function will use the current node positions.
            pos_scale (float): The scaling factor for the node positions. Default is 1.
            node_color (str or list): The color(s) of the nodes.
            node_size (int or list): The size(s) of the nodes.
            with_labels (bool): Whether to display the node labels. Default is True.
            font_size (int): The font size of the node labels.
            font_color (str): The color of the node labels. Default is None.
            edge_colors (str or list, optional): The color(s) of the edges. If None, defaults of plotly will be used.
            edge_widths (int or list, optional): The width(s) of the edges. If None, defaults of plotly will be used.
            legend_config (list): A list of tuples containing the color and label for each legend item.
            legend_fontsize (int): The font size for legend elements.
            figsize (tuple): The size of the figure in inches (width, height). Default is None.
            bgcolor (str): The background color of the plot. Default is 'moccasin'.

        Returns:
            plotly.graph_objects.Figure: The plotly figure object.
        """
        if graph is None:
            graph = self
        if pos_scale is None:
            pos_scale = 1
        if not output_path.endswith('.html'):
            output_path = f'{os.path.splitext(output_path)[0]}.html'
        
        # import and get degree list for caption
        import plotly.graph_objects as go
        degrees = nx.degree(graph)

        dpi = plt.rcParams['figure.dpi']
        height = width = None
        if figsize:
            height = figsize[1] * dpi
            width = figsize[0] * dpi
        plotly_fig = go.Figure(
             layout=go.Layout(
                autosize=False,
                width=width,
                height=height,
                showlegend=legend_config is not None,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                legend_font={'family': 'Arial', 'size': legend_fontsize, 'color': None},
                hovermode='closest',
                margin=dict(b=10, l=5, r=5, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=bgcolor, 
                paper_bgcolor='lightgray',
             )
        )
        
        existing_edge_x = []
        existing_edge_y = []
        existing_edge_color = edge_colors
        existing_edge_width = edge_widths
        active_edge_x = []
        active_edge_y = []

        for e, (source, target) in enumerate(graph.edges):
            x0, y0 = pos[source] * pos_scale
            x1, y1 = pos[target] * pos_scale
            # establishing whether the edge is active via comparing the widh
            if self.is_dynamic and edge_widths[e] == ACTIVE_EDGE_WIDTH:
                edge_x, edge_y = active_edge_x, active_edge_y
            else:
                edge_x, edge_y = existing_edge_x, existing_edge_y
                if self.is_dynamic:
                    existing_edge_color, existing_edge_width = edge_colors[e], edge_widths[e]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        if existing_edge_x:
            plotly_fig.add_trace(go.Scatter(
                x=existing_edge_x, y=existing_edge_y,
                line=dict(width=existing_edge_width, color=existing_edge_color),
                hoverinfo='none',
                mode='lines',
                name='Edges',
            ))
        if active_edge_x:
            plotly_fig.add_trace(go.Scatter(
                x=active_edge_x, y=active_edge_y,
                line=dict(width=ACTIVE_EDGE_WIDTH, color=ACTIVE_EDGE_COLOR),
                hoverinfo='none',
                mode='lines',
                name='Active Edges',
            ))
        
        # Add nodes to figure
        node_x = []
        node_y = []
        for node in graph.nodes:
            x, y = pos[node] * pos_scale
            node_x.append(x)
            node_y.append(y)
            if with_labels:
                plotly_fig.add_annotation({
                    'align': 'center',
                    'font': {'family': 'Arial', 'size': font_size if font_size is not None else node_size / 1.5, 'color': font_color},
                    'opacity': 1,
                    'showarrow': False,
                    'text': f'<b>{node}</b>',
                    'x': x,
                    'xanchor': 'center',
                    'xref': 'x',
                    'y': y,
                    'yanchor': 'middle',
                    'yref': 'y'
                })
                
        # separate logic for when nodes have labels and when they don't    
        if legend_config is None:
            plotly_fig.add_trace(go.Scatter(
                name='Nodes',
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color=node_color,
                    size=node_size,
                    textfont=dict(size=legend_fontsize, color=font_color),
                    line_width=1),
                text=[f'Node: {node}<br />Class: Unknown<br />Degree: {degree}' \
                      for node, degree in degrees],
            ))
        else:
            # we need to convert these lists to numpy arrays in order to select multiple indexes for legend purposes
            node_color = np.array(node_color) if isinstance(node_color, Iterable) and not isinstance(node_color, str) else node_color
            node_size = np.array(node_size) if isinstance(node_size, Iterable) else node_size
            node_x = np.array(node_x)
            node_y = np.array(node_y)
            degrees = np.array(degrees)
            for legend_color, legend_label in legend_config:
                # look in the 'node_color' collection for the indexes where the current 'legend_color' can be found
                indexes = np.nonzero(node_color == legend_color)[0]
                plotly_fig.add_trace(go.Scatter(
                    name=legend_label,
                    x=node_x[indexes], y=node_y[indexes],
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        color=legend_color,
                        size=node_size[indexes] if isinstance(node_size, np.ndarray) else node_size,
                        line_width=1),
                    text=[f'Node: {node}<br />Class: {legend_label}<br />Degree: {degree}' \
                          for node, degree in degrees[indexes]],
                ))
            
        if not notebook:
            plotly_fig.write_html(output_path)

        return plotly_fig
    
    def draw_pyvis(self, graph=None, output_path='fig/graph.html', notebook=True, pos=None, pos_scale=200, node_color=None, node_size=10, with_labels=True, font_size=10, 
                   font_color='white', edge_colors=None, edge_widths=None, legend_config=None, legend_fontsize=15, figsize=None, bgcolor='moccasin', show_buttons=False, 
                   show_physics_buttons=True, pyvis_options=None):
        """
        This function accepts a networkx graph object, converts it to a pyvis network object preserving its node and edge attributes,
        and both returns and saves a dynamic network visualization.
        Valid node attributes include:
            "size", "value", "title", "x", "y", "label", "color".
            (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)
        Valid edge attributes include:
            "arrowStrikethrough", "hidden", "physics", "title", "value", "width"
            (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)

        Args:
            graph (networkx.Graph): The graph to generate the layout for. If None, uses the self instance.
                This parameter permits drawing a different graph from self (e.g. a subgraph of active edges if self is dynamic).
            output_path (str): The path of the output file. Default is 'fig/graph.html'.
            notebook (bool): Whether to display the graph in a Jupyter notebook. Default is True.
            pos (dict): A dictionary of node positions. If None, the function will use the current node positions.
            pos_scale (float): The scaling factor for the node positions. Default is 200.
            node_color (str or list): The color(s) of the nodes.
            node_size (int or list): The size(s) of the nodes.
            with_labels (bool): Whether to display the node labels. Default is True.
            font_size (int): The font size of the node labels.
            font_color (str): The color of the node labels. Default is None.
            edge_colors (str or list): The color(s) of the edges.
            edge_widths (int or list): The width(s) of the edges.
            legend_config (list): A list of tuples containing the color and label for each legend item.
            legend_fontsize (int): The font size for legend elements.
            figsize (tuple): The size of the figure in inches (width, height). Default is None.
            bgcolor (str): The background color of the plot. Default is 'moccasin'.
            show_buttons (bool): Show buttons in saved version of network?
            show_physics_buttons (bool): Show only buttons controlling physics of network?
            pyvis_options (str): Provide pyvis-specific options in a JSON-like format 
                More details: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.options.Options.set.

        Returns:
            pyvis.network.Network: the converted pyvis network object.
        """
        if graph is None:
            graph = self
        if pos_scale is None:
            pos_scale = 200
        if not output_path.endswith('.html'):
            output_path = f'{os.path.splitext(output_path)[0]}.html'

        # local import and get degree list for caption
        from pyvis import network as pyvis_net
        degrees = nx.degree(graph)
        
        if self.pyvis_graph is None:
            dpi = plt.rcParams['figure.dpi']
            height = width = None
            if figsize:
                height = figsize[1] * dpi
                width = figsize[0] * dpi
            network_class_parameters = {'notebook': notebook, 'height': height, 'width': width, 'bgcolor': bgcolor, 'cdn_resources': 'remote'}
            # make a pyvis graph
            self.pyvis_graph = pyvis_net.Network(**{parameter_name: parameter_value for parameter_name, parameter_value in network_class_parameters.items() if parameter_value})

            max_x = 0
            min_y = 1e5
            # for each node and its attributes in the networkx graph
            for i, node in enumerate(graph.nodes):
                label = str(self.node_names[node] if self.node_names else i)
                node_attrs = {
                    'title': f'Node {label}\nDegree: {degrees[node]}',
                    'color': node_color[node] if node_color else 'green',
                    'label': label if with_labels else None,
                    'size': node_size[i] if isinstance(node_size, Iterable) else node_size,
                    'shape': 'circle',
                    'font': {'size': font_size, 'color': font_color},
                }
                if pos:
                    node_attrs['x'] = pos[i][0] * pos_scale
                    node_attrs['y'] = pos[i][1] * pos_scale
                    max_x = max(max_x, node_attrs['x'])
                    min_y = min(min_y, node_attrs['x'])
                # add the node
                self.pyvis_graph.add_node(node, **node_attrs)

            # for each edge and its attributes in the networkx graph
            for e, (source, target, edge_attrs) in enumerate(graph.edges(data=True)):
                edge_attrs = {
                    'title': f'Edge {source}-{target} \nWeight: {"%.2f" % edge_attrs["weight"]}',
                    'color': edge_colors[e] if isinstance(edge_colors, Iterable) and not isinstance(edge_colors, str) else edge_colors,
                    'width': edge_widths[e] if isinstance(edge_widths, Iterable) else edge_widths
                }
                # add the edge
                self.pyvis_graph.add_edge(source, target, **edge_attrs)

            if legend_config:
                # access one element from legend_config to determine whether the labels are fullnames or letter abbrv
                full_name = len(legend_config[0][1]) > 1
                # Add Legend Nodes
                step = 70 if full_name else 50
                x = max_x + 120 if pos else -500
                y = min_y - 120 if pos else -500
                num_actual_nodes = max(self)
                
                legend_id = 0
                for legend_label in [f'Control Day: {self.control_iter}', f'Testing: {self.tested}', f'Tracing: {self.traced}']:
                    legend_id += 1
                    self.pyvis_graph.add_node(num_actual_nodes + legend_id, **{
                                # 'group': legend_id,
                                'color': 'darkgray',
                                'label': legend_label,
                                'size': 30, 
                                # 'fixed': True,
                                'physics': False, 
                                'x': x, 
                                'y': f'{y + legend_id*step}px',
                                'shape': 'box', 
                                'widthConstraint': legend_fontsize * 15, 
                                'font': {'size': legend_fontsize, 'color': font_color, 'weight': 600}
                    })
                for legend_color, legend_label in legend_config:
                    legend_id += 1
                    self.pyvis_graph.add_node(num_actual_nodes + legend_id, **{
                            # 'group': legend_id,
                            'color': legend_color,
                            'label': legend_label,
                            'size': 30, 
                            # 'fixed': True,
                            'physics': False, 
                            'x': x, 
                            'y': f'{y + legend_id*step}px',
                            'shape': 'box', 
                            'widthConstraint': 200 if full_name else 50, 
                            'font': {'size': legend_fontsize, 'color': font_color, 'weight': 600}
                    })

            self.pyvis_graph.toggle_physics(False)
            # turn buttons on
            if show_buttons:
                self.pyvis_graph.show_buttons()
            elif show_physics_buttons:
                self.pyvis_graph.show_buttons(filter_=['physics'])
            # pyvis-specific options
            if pyvis_options:
                self.pyvis_graph.set_options(pyvis_options)
        
        else:
            # self.pyvis_graph.heading = f'Control Day: {self.control_iter}' can be used as heading
            for node in self.pyvis_graph.nodes:
                try:
                    node['color'] = node_color[int(node['label'])] if node_color else 'green'
                except ValueError:
                    if 'Test' in node['label']:
                        node['label'] = f'Testing: {self.tested}'
                    elif 'Trac' in node['label']:
                        node['label'] = f'Tracing: {self.traced}'
                    elif 'Day' in node['label']:
                        node['label'] = f'Control Day: {self.control_iter}'
                        
            if self.is_dynamic:
                for e, edge in enumerate(self.pyvis_graph.get_edges()):
                    edge['color'] = edge_colors[e]
                    edge['width'] = edge_widths[e]

        # return and also save
        return self.pyvis_graph.show(output_path)

    
###
# Network generator functions:
#   - get_random - returns a random Network obj with links created according to type 'typ'
#   - get_from_predef - returns a Network obj with links matching a predefined nx.Graph
#   - get_dual - a tracing subview Network obj is created from an input Network
#   - get_dual_from_predef - a tracing subview Network obj is created from an input nx.Graph
###
    
def get_random(netsize=200, k=10, typ='random', rem_orphans=False, weighted=False, contact_reduction=0, mask_uptake=0, mask_cut=.5, sah_uptake=0, p=.1, count_importance=1, 
               nseed=None, inet=0, use_weights=False, edge_sample_size=None, edge_prob_sampling=True, is_dynamic=False, **kwargs):
    """
    Generates a random network with the specified parameters.

    Args:
        netsize (int, optional): The number of nodes in the network (default is 200).
        k (int, optional): The average degree or the `m` parameter used in the network model, depending on the type of the latter (default is 10).
        typ (str, optional): The type of random network to generate (default is 'random'). Options are described in `init_random`.
        rem_orphans (bool, optional): Whether to remove orphan nodes (default is False).
        weighted (bool, optional): Whether to create graph with weighted edges (default is False).
        contact_reduction (float, optional): The percentage reduction in contacts due to interventions (default is 0).
        mask_uptake (float, optional): The percentage of nodes using masks (default is 0).
        mask_cut (float, optional): The mask-induced reduction in the spread (default is 0.5).
        sah_uptake (float, optional): The percentage of nodes under stay-at-home orders (default is 0).
        p (float, optional): The probability of adding an edge for each node (default is 0.1).
        count_importance (int, optional): The importance of neighbor counts, used to scale the total number of neighbors for infection/tracing diffusion (default is 1).
        nseed (int, optional): The seed for the random number generator (default is None).
        inet (int, optional): The index of the network (default is 0).
        use_weights (bool, optional): Whether to use weights for the calculation of neighbor counts (default is False).
        edge_sample_size (int, optional): The number/percentage of edges to sample (default is None).
        edge_prob_sampling (bool, optional): Whether to sample edges probabilistically, where edges are selected with probability `edge_sample_size`,
            or deterministically, where a fraction of `edge_sample_size` edges from the total edge list are selected at random.
        is_dynamic (bool, optional): Whether the Network changes edges dynamically (default is False).
        **kwargs (dict, optional): Additional keyword arguments. Used to allow all network generator functions to have the same signature.

    Returns:
        G (Network): A random network with the specified parameters.
    """
    G = Network()
    # give this network an index
    G.inet = inet
    # set whether neighbor counts use weights
    G.use_weights = use_weights
    # attach a neighbor count importance to this network
    G.count_importance = count_importance
    # whether edges will be dynamic
    G.is_dynamic = is_dynamic
    # for interventions like partial lockdowns
    G.contact_reduction = contact_reduction
    
    rand = np.random.RandomState(nseed)
    # uptake of masks and stay-at-home orders
    if mask_uptake:
        mask_uptake = int(mask_uptake if mask_uptake >= 1 else mask_uptake * netsize)
        G.node_mask = set(rand.choice(range(netsize), mask_uptake, replace=False))
        G.mask_cut = mask_cut
    if sah_uptake:
        sah_uptake = int(sah_uptake if sah_uptake >= 1 else sah_uptake * netsize)
        G.node_sah = rand.choice(range(netsize), sah_uptake, replace=False)
        
    # initialize random conections
    G.init_random(netsize, k, typ=typ, p=p, weighted=weighted, seed=nseed, edge_sample_size=edge_sample_size, edge_prob_sampling=edge_prob_sampling)
    # Set the active node list (i.e. without orphans if rem_orphans=True)
    G.node_list = list(G)
    if rem_orphans:
        for nid in G.nodes:
            if G.degree(nid) == 0:
                G.node_list.remove(nid)
                # we mark orphans as isolated (even if NOT technically traced, this will not impact the outcome since orphans have no neighbors)
                # Note that the node_traced list will be copied over to the tracing network!
                G.node_traced[nid] = True
    return G


def get_from_predef(nx_or_edgelist, rem_orphans=False, ids=None, count_importance=1, nseed=None, inet=0, use_weights=False, contact_reduction=0, mask_uptake=0, mask_cut=.5, 
                    sah_uptake=0, W_factor=0, K_factor=10, reindex=False, **kwargs):
    """
    Returns a Network object generated from a pre-defined network.

    Args:
        nx_or_edgelist (networkx.Graph or list): A networkx graph or a list of edges.
        rem_orphans (bool, optional): Whether to remove orphan nodes (default in False).
        ids (list, optional): A list of supplied node IDs to use for the network (default is None). These are used only if `nx_or_edgelist` is NOT a nx graph.
            If `ids` is empty (and `nx_or_edgelist` is a list of edges), the node IDs are inferred from the edge list.
        count_importance (int, optional): The importance of neighbor counts, used to scale the total number of neighbors for infection/tracing diffusion (default is 1).
        nseed (int, optional): Seed for the random number generator (Default is None).
        inet (int, optional): Index for the network (default is 0).
        use_weights (bool, optional): Whether to use weights for the calculation of neighbor counts (default is False).
        contact_reduction (float, optional): Contact reduction for interventions like partial lockdowns (default is 0).
        mask_uptake (float, optional): The percentage of nodes using masks (default is 0).
        mask_cut (float, optional): The mask-induced reduction in the spread (default is 0.5).
        sah_uptake (float, optional): The percentage of nodes under stay-at-home orders (default is 0).
        W_factor (int, optional): Denominator of the normalization factor for edge weights (default is 0, where the `norm_factor` is set to 1).
        K_factor (int, optional): Numerator of the normalization factor for edge weights (default is 10).
        reindex (bool, optional): Whether to reindex the node IDs - i.e. make ID range (0,netsize) (default is False).
        **kwargs: Additional keyword arguments. Used to allow all network generator functions to have the same signature.

    Returns:
        G (Network): A random network with the specified parameters.

    Notes:
        - If a networkx graph is passed, the node IDs are inferred from the graph.
        - If a list of edges is passed, the node IDs are inferred from the edge list.
        - If a list of edges is passed, the node IDs can be supplied in the optional argument 'nettype' as a dictionary with a 'nid' key.
        - If norm_factor = K_factor / W_factor is <= 1, the edge weights are not normalized.
    """
    G = Network()
    # give this network an index
    G.inet = inet
    # set whether neighbor counts use weights
    G.use_weights = use_weights
    # give this network a count_importance -> used to supply different tau rates
    G.count_importance = count_importance
    # for interventions like partial lockdowns
    G.contact_reduction = contact_reduction
    # normalization factor for edges
    G.norm_factor = K_factor / W_factor if W_factor else 1
    # Try to access fields based on nx API. If this fails, assume only a list of edges was supplied in nx_or_edgelist
    try:
        ids = nx_or_edgelist.nodes
        edges = list(nx_or_edgelist.edges(data=True))
    except AttributeError:
        # if user has not supplied node ids, infer the nodes from the edge list
        if not ids:
            # Need to obtain the set of node ids from the edge list
            edge_arr = np.array(nx_or_edgelist, dtype=object)
            ids = set(edge_arr[:, 0]).union(set(edge_arr[:, 1]))
        # edges will be the unmodified nx_or_edgelist argument
        edges = nx_or_edgelist
    if reindex:
        new_ids = range(len(ids))
        G.node_names = dict(zip(new_ids, ids))
        G.names_to_node = dict(zip(ids, new_ids))
        ids = new_ids
        
    # add the ids of nodes to the network
    G.add_mult(ids=ids, state='S', traced=False)
    # copy over already generated drawing layout, if one exists
    try:
        G.pos = nx_or_edgelist.pos
    except AttributeError:
        pass
    rand = np.random.RandomState(nseed)
    # uptake of masks and stay-at-home orders
    if mask_uptake:
        mask_uptake = int(mask_uptake if mask_uptake >= 1 else mask_uptake * len(ids))
        G.node_mask = set(rand.choice(G.nodes, mask_uptake, replace=False))
        G.mask_cut = mask_cut
    if sah_uptake:
        sah_uptake = int(sah_uptake if sah_uptake >= 1 else sah_uptake * len(ids))
        G.node_sah = rand.choice(G.nodes, sah_uptake, replace=False)
        
    # checking if there are any edges; agnostic for python iterables and np.arrays
    if is_not_empty(edges):
        # supply edges but do not update counts yet since the first infected have not been set by this point
        G.add_links(edges, update=False, reindex=reindex)
        # record first set of edges
        G.first_edges = list(G.edges(data=True))

    # Set the active node list (i.e. without orphans if rem_orphans=True)
    G.node_list = list(G)
    if rem_orphans:
        for nid in G.nodes:
            if G.degree(nid) == 0:
                G.node_list.remove(nid)
                # we mark orphans as isolated (even if NOT technically traced, this will not impact the outcome since orphans have no neighbors)
                # Note that the node_traced list will be copied over to the tracing network!
                G.node_traced[nid] = True
    return G


def get_dual(G, net_overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, conserve_overlap=True, count_importance=1, nseed=None, active_based_noising=False, **kwargs):
    """
    Returns a copy of the input graph with noise added to its links, marked as a dual network. This can be used to generate a tracing subview of the input network.

    Args:
        G (networkx.Graph): input graph.
        net_overlap (float, optional): overlap between the two networks.
        z_add (float, optional): probability of adding a link.
        z_rem (float, optional): probability of removing a link.
        keep_nodes_percent (float, optional): percentage of nodes to keep in the tracing subview (i.e. uptake).
        conserve_overlap (bool, optional): whether to conserve overlap between the two networks.
        count_importance (int, optional): importance of neighborhood node counts (i.e. tau_t).
        nseed (int, optional): seed for random number generator.
        active_based_noising (bool, optional): whether to calculate the fraction of noise links based on active nonorphan nodes only.
        **kwargs: Additional keyword arguments. Used to allow all network generator functions to have the same signature.

    Returns:
        D (networkx.Graph): copy of the input graph with noise added to its links, marked as a dual network
    """
    # the copy of the graph will include everything at this point, including active node_list, node_states, node_traced and node_counts
    # it will also have a SEPARATE entity for node_counts
    D = G.copy()
    # mark this net as dual
    D.is_dual = True
    # set the importance
    D.count_importance = count_importance
    # noise the links according to the parameters given
    D.noising_links(net_overlap, z_add, z_rem, keep_nodes_percent, conserve_overlap, update=True, seed=nseed, active_based_noising=active_based_noising)
    return D


def get_dual_from_predef(G, nx_or_edgelist, count_importance=1, W_factor=0, K_factor=10, reindex=False, **kwargs):
    """
    Returns a Network object generated from a pre-defined network, marked as a dual network. This can be used to generate a tracing subview of the input network.

    Args:
        G (networkx.Graph): input graph.
        nx_or_edgelist (networkx.Graph or list): A networkx graph or a list of edges to be used for the generation of the dual view.
        count_importance (int, optional): importance of neighborhood node counts (i.e. tau_t).
        W_factor (int, optional): Denominator of the normalization factor for edge weights (default is 0, where the `norm_factor` is set to 1).
        K_factor (int, optional): Numerator of the normalization factor for edge weights (default is 10).
        reindex (bool, optional): Whether to reindex the node IDs - i.e. make ID range (0,netsize) (default is False).
        **kwargs: Additional keyword arguments. Used to allow all network generator functions to have the same signature.

    Returns:
        D (networkx.Graph): copy of the input graph with noise added to its links, marked as a dual network.

    Notes:
        - The node IDs are mantained from the input graph G.
        - The edges are added to the dual copy according to the input nx_or_edgelist.
        - If norm_factor = K_factor / W_factor is <= 1, the edge weights are not normalized.
    """
    # the copy of the graph will include everything at this point, including active node_list, node_states, node_traced
    # it will also have a SEPARATE entity for node_counts
    D = G.copy()
    # mark this net as dual
    D.is_dual = True
    # give this network a count_importance -> used to supply different tau rates
    D.count_importance = count_importance
    # used for normalization of weights
    D.norm_factor = K_factor / W_factor if W_factor else 1
    # Try to access fields of nx_or_edgelist based on nx API. If this fails, assume only a list of edges was supplied in nx_or_edgelist
    try:
        edges = list(nx_or_edgelist.edges(data=True))
    except AttributeError:
        edges = nx_or_edgelist
    if reindex:
        new_ids = range(len(ids))
        G.node_names = dict(zip(new_ids, ids))
        G.names_to_node = dict(zip(ids, new_ids))
        ids = new_ids
    # clear the true network edges since we want to put the predefined set in
    D.clear_edges()
    # supply edges and norm_factor weight normalization factor
    D.add_links(edges, update=True)
    return D


def configuration_model(deg_sequence, seed=0):
    """
    Generates a random graph using the configuration model algorithm. Compared to the original networkx function, self-loops and parallel edges are also removed.

    Args:
        deg_sequence (list): A list of integers representing the degree sequence of the graph.
        seed (int, optional): Seed for the random number generator (default is 0).

    Returns:
        G (networkx.Graph): A random networkx graph generated using the configuration model algorithm.
    """
    G = nx.configuration_model(deg_sequence, create_using=nx.Graph, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G