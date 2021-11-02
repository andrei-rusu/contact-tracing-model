import networkx as nx
import inspect
import itertools
import random
from sys import stderr

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lib.utils import rand_pairs, rand_pairs_excluding, get_z_for_overlap, get_overlap_for_z
from lib.simulation import Simulation
from lib.utils import is_not_empty

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
    'S': 'darkgreen',
    'E': 'yellow',
    'I': 'orange',
    'Ip': 'orange',
    'Ia': 'pink',
    'Is': 'red',
    'H': 'cyan',
    'T': 'blue',
    'R': 'lime',
    'D': 'gray',
    'N': 'purple'
}

MODEL_TO_STATES = {
    'sir': ['S', 'I', 'R', 'T', 'N'],
    'seir': ['S', 'E', 'I', 'R', 'T', 'N'],
    'covid': ['S', 'E', 'Ip', 'Ia', 'Is', 'H', 'D', 'R', 'T', 'N']
}


class Network(nx.Graph):
    
    def get_simulator(self, *args, **kwargs):
        return Simulation(self, *args, **kwargs)
    
    def __init__(self, *args, **kwargs):
        # initialize the network id to 0
        self.inet = 0
        # set use_weights default - whether or not weights of edges are considered
        self.use_weights = False
        # Atomic Counter for node ids
        self.cont = itertools.count()
        # Overlap after noising - initially perfect overlap, MAY change after calling 'noising_links'
        self.overlap = 1
        # Maintain 'active' (not orphans) node list + neighbor counts; and inf + traced states for all nodes
        self.node_list = []
        self.node_states = []
        self.node_traced = []
        # nids -> State -> neighbor counts in State
        self.node_counts = defaultdict(lambda: defaultdict(float))
        self.traced_time = {}
        self.noncomp_time = {}
        
        ### Vars that are not copied over by the copy method
        # whether this is an original or a dual network - this is needed when drawing
        self.is_dual = False
        # relative counts importance (will be multiplied with the counts to obtain their importance in the rate)
        self.count_importance = 1
        # Initialize pos to None - a new spring layout is created when first drawn
        self.pos = None
        # Normalization term used in weight normalization
        self.norm_factor = 0
        super().__init__(*args, **kwargs)
        
    def copy(self):
        # relying on the basic copying mechanism of networkx
        obj = super().copy()
        # use copy_state_from to copy over extra state information from the true net to the new network
        obj.copy_state_from(self)
        return obj
    
    def copy_state_from(self, obj):
        # instance primitives
        self.inet = obj.inet
        self.use_weights = obj.use_weights
        self.norm_factor = obj.norm_factor
        # lists or dict which will be SHARED between true net and dual net
        self.cont = obj.cont
        self.node_list = obj.node_list
        self.node_states = obj.node_states
        self.node_traced = obj.node_traced
        self.traced_time = obj.traced_time
        self.noncomp_time = obj.noncomp_time
        # node_counts is the only reference not copied over from the original network
        self.node_counts = defaultdict(lambda: defaultdict(float))

        
    def init_random(self, n=200, k=10, typ='random', p=.1, weighted=False, seed=None):
        # add n nodes to the network in the start state -> S
        self.add_mult(n, state='S', traced=False)
        # initialize the rewire probability to k/n if p=None provided
        p = k / n if p is None else p
                
        links_to_create_dict = {
            # Generate random pairs (edges) without replacement
            'random': lambda: rand_pairs(n, int(n * k / 2), seed=seed),
            'binomial': lambda: list(nx.fast_gnp_random_graph(n, p, seed=seed).edges),
            # small-world network
            'ws': lambda: list(nx.watts_strogatz_graph(n, k, p, seed=seed).edges),
            'newman-ws': lambda: list(nx.newman_watts_strogatz_graph(n, k, p, seed=seed).edges),
            # scale-free network
            'barabasi': lambda: list(nx.barabasi_albert_graph(n, m=k, seed=seed).edges),
            'powerlaw-cluster': lambda: list(nx.powerlaw_cluster_graph(n, m=k, p=p, seed=seed).edges),
            # fully connected network
            'complete': lambda: list(nx.complete_graph(n).edges),
        }
        try:
            links_to_create = links_to_create_dict[typ]()
        except KeyError:
            print("The inputted network type is not supported. Default to: random", file=stderr)
            links_to_create = links_to_create_dict['random']()
            
        len_links = len(links_to_create)
            
        if weighted:
            # Create random weights from 1 to 10 and append them to the list of edge tuples
            weights = random.choices(range(1,10), k=len_links)
            links_to_create = [links_to_create[i] + (weights[i],) for i in range(len_links)]
            
        # Add the random edges with/without weights depending on 'weighted' parameter
        # we do not update the counts here because they will be updated after the infection has been seeded with first_inf
        self.add_links(links_to_create, update=False)
        
    def noising_links(self, overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, conserve_overlap=True, update=True, seed=None, active_based_noising=False):
        # Recover the number of nodes for which noising links operations will be performed
        # Note that 'active_based_noising' parameter can be used to change between using self.nodes (all nodes) or self.node_list (only active nonorphan nodes)
        nodes = self.node_list if active_based_noising else self.nodes
        n = len(nodes)
        # Recover average degree
        k = self.avg_degree()
        
        # Current edge set (prior to noising, nodes in each edge tuple are sorted)
        current_edges = {tuple(sorted(edge)) for edge in self.edges}
        
        # links to add and rem lists
        links_to_add, links_to_rem = [], []
        remaining_links_rem = 0
        
        # seed local random if available
        rand = random if seed is None else random.Random(seed)
        
        # If we are interested in maintaining the overlap, then the correct z_add and z_rem will be calculated based on the params
        if conserve_overlap:
            # If no overlap value, use z_add & z_rem
            if overlap is None or overlap == -1:
                # Average degree k, z_add, z_rem used to calculate overlap after noising
                self.overlap = get_overlap_for_z(k, z_add, z_rem)
            # If overlap given, calculate z_add and z_rem; SUPPLIED z_add and z_rem are mostly IGNORED
            else:
                self.overlap = overlap
                # if z_add = 0, z_add will stay 0 while z_rem = z; OTHERWISE z_add = z_rem = z
                z_add, z_rem = get_z_for_overlap(k, overlap, include_add = z_add)
                
            # Random adding z_add on average - this should be 0 in case of digital tracing networks
            remaining_links_add = int(n * z_add / 2)
            links_to_add = rand_pairs_excluding(n, remaining_links_add, to_exclude=current_edges, seed=seed) if z_add else []

            # Random removing z_rem on average - this will also cover the case keep_nodes_percent < 1
            remaining_links_rem = int(n * z_rem / 2)
            #
            # The logic for removing edges is done below (need to account for keep_nodes_percent value)
        
        
        # Simple case, no node's edges are removed completely from the network unless the randomness does it
        if keep_nodes_percent == 1:
            links_to_rem = rand.sample(current_edges, remaining_links_rem)
            
        # The links of certain nodes will be completely removed if keep_nodes_percent is below 1 (<100%)
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
            # in the case the supplied overlap gets ultimately ignored, a new value needs to be calculated for it
            else:
                actual_z_add = 0
                actual_z_rem = len_untraceable_edges * 2 / n
                self.overlap = get_overlap_for_z(k, actual_z_add, actual_z_rem)

                
        self.add_edges_from(links_to_add)
        self.remove_edges_from(links_to_rem)

        # initially there is no one traced that we care about, so we only update counts for infection status
        if update:
            self.update_counts()
    
    def init_for_simulation(self, first_inf_nodes):
        # this will initialize states to 'S' and tracing status to False
        self.init_states()
        # initially, there should be no one traced so we only update the infection status counts
        self.change_state(first_inf_nodes, state='I', update=True, update_with_traced=False)
    
    def init_states(self, state='S'):
        # Set all nodes back to 'state' and traced=False
        len_to_max_id = max(self) + 1
        list_to_max_id = range(len_to_max_id)
        self.node_states = [state] * len_to_max_id
        # the tracing states array must reflect that only the active nodes are traceble
        self.node_traced = np.isin(list_to_max_id, self.node_list, invert=True).tolist()
        self.traced_time = {}
        self.noncomp_time = {}

    def add(self, state='S', traced=False, ids=None, update=False):
        """
        The atomic counter gets incremented only if no predefined ids is supplied.
        This method will attempt to add a node no matter what is already in the network
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
            self.node_traced += [False] * needed_len_state_list
        # finally amend the 'state' and 'traced' status of this added node in the lists
        self.node_states[ids] = state
        self.node_traced[ids] = traced
        
    def add_mult(self, n=200, state='S', traced=False, ids=None, update_existing=False):
        """
        If ids list is supplied, it will be given priority, so n will be effectively ignored
        Tf ids is not supplied, this method does guarantees n additions will be performed past the max id or current counter
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
            self.node_traced += [False] * needed_len_state_list
        # populate lists with correct state and traced status for ids in question
        for i in ids:
            # update_existing = True -> all nodes get 'state' and 'traced' status from the parameters
            # otherwise: check if node was NOT in the set of active nodes prior to this update (note self.node_list not yet updated here)
            if update_existing or i not in self.node_list:
                self.node_states[i] = state
                self.node_traced[i] = traced

    def update_counts(self, nlist=None):
        # local for efficiency
        counts = self.node_counts
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
        # local for efficiency
        counts = self.node_counts
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
        return self.node_counts[nid][state]
        
    def get_state(self, nid):
        return self.node_states[nid]
        
    def change_state(self, nids, state='I', update=True, update_with_traced=False):
        for nid in np.atleast_1d(nids):
            self.node_states[nid] = state
        if update:
            self.update_counts_with_traced() if update_with_traced else self.update_counts()
                
    def change_state_fast_update(self, nid, state):
        # local vars for efficiency
        states = self.node_states
        counts = self.node_counts
        # remember old state
        old_state = states[nid]
        
        # update neighbor counts if NOT moving to Exposed (We care only about INFECTIOUS transitions)
        # update neighbor counts only if NOT traced / NOT hospitalized CURRENTLY
        if state != 'E' and not (self.node_traced[nid] or old_state == 'H'):
            for _, neigh, data in self.edges(nid, data=True):
                # update counts only for Suscpetibles or if the state transitions correspond to a tracing event
                if states[neigh] == 'S' or state == 'T' or state=='N':
                    # get the counts dict of the neighbor
                    neigh_counts = counts[neigh]
                    # count_val becomes the normalized weight of this edge if self.use_weight
                    count_val = data['weight'] if self.use_weights else 1
                    neigh_counts[old_state] -= count_val
                    neigh_counts[state] += count_val
                    
        # update the actual node state
        states[nid] = state
    
    def change_traced_state_update_tracing(self, nid, to_traced, time_of_trace=None, legal_isolation_exit=False):
        """
        When using separate_traced, the T state is independent of all the other states and represented via a flag
        Counts are updated as-if this would be an actual state change: 
         - node_counts for neighbor node -> for current inf state of nid decrease/increase, for 'T' increment/decremenet
         based on to_traced True/False
        """
        # a change happens only when its a valid change of traced state
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
               
    def change_traced_state_update_infectious(self, nid, to_traced, time_of_trace=None, legal_isolation_exit=False):
        """
        When using separate_traced, the T state is independent of all the other states and represented via a flag
        Counts are updated as-if this would be an actual state change: 
        - node_counts for neighbor node -> for current inf state of nid decrease/increase, for 'T' increment/decremenet
        Whichever of the above, depends directly on to_traced True/False
        - this is meant to be run only on the infection network AND only if the infection status of nid is Infectious
        """
        inf_state = self.node_states[nid]
        if inf_state in ['I', 'Ia', 'Is']:
            # If this is a traced event, count_val = 1 | edge_weight, update time of tracing
            count_val = 1 if to_traced else -1
            # local vars for efficiency
            counts = self.node_counts
            # get this node's current infection network state
            for _, neigh, data in self.edges(nid, data=True):
                # final_count_val becomes the normalized weight of this edge if self.use_weight, with sign dictated by count_val
                counts[neigh][inf_state] -= count_val * data['weight'] if self.use_weights else count_val

                
    def add_link(self, nid1, nid2, weight=1, update=False, update_with_traced=False):
        self.add_edge(nid1, nid2, weight=weight)
        if update:
            self.update_counts_with_traced([nid1, nid2]) if update_with_traced else self.update_counts([nid1, nid2])
            
    def rem_link(self, nid1, nid2, update=False, update_with_traced=False):
        self.remove_edge(nid1, nid2)
        if update:
            self.update_counts_with_traced([nid1, nid2]) if update_with_traced else self.update_counts([nid1, nid2])
            
    def add_links(self, lst, weight=1, update=False, update_with_traced=False):
        """
        Note, with norm_factor<=0, normalization of weights is cancelled and the edges will be added to the graph as-is
        """
        # if edges with no weights are to be added, turn off use_weights flag
        if len(lst[0]) == 2:
            self.use_weights = False
        # norm_factor used for normalizing weights
        norm_factor = self.norm_factor
        # If the use of weights is turned off or inappropriate norm_factor supplied, add edges in whichever way works given 'lst' and 'weight' without normalization
        if not self.use_weights or norm_factor <= 0:
            try:
                self.add_edges_from(lst, weight=weight)
            except TypeError:
                self.add_weighted_edges_from(lst)
        # If we care about the weights, and an appropriate norm_factor exists, normalize weights and add edges accordingly
        else:
            # If a sensible norm_factor is supplied, weighted averaging of the edge weights is carried out
            # We sill allow either the form [1,2,3] or [1,2, {weight:3}]
            try:
                edges_arr = [(n1, n2, data['weight'] * norm_factor) for n1, n2, data in lst]
            except (TypeError, IndexError):
                # the array of edges will hold both ints (nids) and floats (weights)
                edges_arr = np.array(lst, dtype=object)
                edges_arr[:, -1] *= norm_factor
            finally:
                self.add_weighted_edges_from(edges_arr)

        if update:
            self.update_counts_with_traced() if update_with_traced else self.update_counts()
        
            
    def avg_degree(self, use_weights=False):
        if use_weights:
            return np.mean(list(dict(self.degree(weight='weight')).values()))
        else:
            return 2 * len(self.edges) / len(self.nodes)
    
    def compute_efforts(self, taur):
        node_states = self.node_states
        node_traced = self.node_traced
        node_counts = self.node_counts
        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        
        # compute random tracing effort -> we only care about the number of traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        randEffortAcum = 0
        # compute active tracing effort -> we only care about the neighs of nodes in traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        tracingEffortAccum = 0
        for nid in self.node_list:
            current_state = node_states[nid]
            if not node_traced[nid] and current_state in traceable_states:
                randEffortAcum += 1
                tracingEffortAccum += node_counts[nid]['T']
                
        return round(taur * randEffortAcum, 2), round(self.count_importance * tracingEffortAccum, 2)
    
        
    def draw(self, pos=None, show=True, ax=None, layout_type='spring_layout', seed=43, legend=True, full_name=False, model='covid', node_size=300, degree_size=False):
        # get the node state list
        node_states = self.node_states
        # for the true network, colors for all nodes are based on their state
        if not self.is_dual:
            colors = [STATES_COLOR_MAP[node_states[nid]] for nid in self.nodes]
        # for the dual network, when no explicit T state set, give priority to node_traced == True as if 'T' state
        # also give priority to node_traced == False as if 'N' state IF the node has been traced before (has a traced_time)
        else:
            colors = []
            for nid in self.nodes:
                if self.node_traced[nid]:
                    color = STATES_COLOR_MAP['T']
                elif nid in self.traced_time:
                    color = STATES_COLOR_MAP['N']
                else:
                    color = STATES_COLOR_MAP[node_states[nid]]
                colors.append(color)
                    
        # sometimes an iterable of axis may be supplied instead of one Axis object, so only get the first element
        if isinstance(ax, Iterable): ax = ax[0]

        # by doing this we avoid overwriting self.pos with pos when we just want a different layout for the drawing
        pos = pos if pos else self.pos
        if pos:
            # if one of self.pos or argument pos was not None, fix the specified node's positions, then generate a new layout around ONLY IF new nodes were added
            if len(self) != len(pos):
                pos = self.generate_layout(pos=pos, fixed=pos.keys(), layout_type=layout_type, seed=seed)
        else:
            # if both were None, generate a new layout with the seed and use it for drawing
            pos = self.generate_layout(layout_type=layout_type, seed=seed)
            
        if degree_size:
            d = nx.degree(self)
            node_size = [(d[node]+1) * 100 for node in self.nodes]
                        
        # draw graph
        nx.draw(self, pos=pos, node_color=colors, ax=ax, with_labels=True, node_size=node_size)
        
        if legend:
            model_states = MODEL_TO_STATES[model]
            if full_name:
                label = lambda state: STATES_NAMES[state]
                plt.subplots_adjust(left=.2)
            else:
                label = lambda state: state
                plt.subplots_adjust(left=.1)
            # create color legend
            plt.legend(handles=[mpatches.Patch(color=STATES_COLOR_MAP[state], label=label(state)) for state in model_states],
                       loc='upper left', prop={'size': 12}, bbox_to_anchor=(0, 1), bbox_transform=plt.gcf().transFigure)
        
        if show:
            plt.show()
            
    def generate_layout(self, layout_type='spring_layout', **kwargs):
        # deals with the case in which the layout was not specified with the correct suffix
        if not layout_type.endswith('_layout'):
            layout_type += '_layout'
        # get the method from networkx which generates the selected layout
        try:
            method_to_call = getattr(nx, layout_type)
        except AttributeError:
            method_to_call = getattr(nx.nx_agraph, layout_type)
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
        self.pos = method_to_call(self, **passed_kwargs)
        return self.pos

    
###
# Graph generator functions:
# get_random - returns a random graph given by typ type
# get_from_predef - load a predefined nxGraph
###
    
    
def get_random(netsize=200, k=10, rem_orphans=False, weighted=False, typ='random', p=.1, count_importance=1, nseed=None, inet=0, use_weights=False, **kwargs):
    G = Network()
    # give this network an index
    G.inet = inet
    # set whether neighbor counts use weights
    G.use_weights = use_weights
    # attach a neighbor count importance to this network
    G.count_importance = count_importance
    # initialize random conections
    G.init_random(netsize, k, typ=typ, p=p, weighted=weighted, seed=nseed)
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


def get_from_predef(nx_or_edgelist, rem_orphans=False, count_importance=1, inet=0, use_weights=False, W_factor=0, K_factor=10, **kwargs):
    """
    Note, 'norm_factor' <= 0 will turn off weight normalization
    """
    G = Network()
    # give this network an index
    G.inet = inet
    # set whether neighbor counts use weights
    G.use_weights = use_weights
    # give this network a count_importance -> used to supply different tau rates
    G.count_importance = count_importance
    # Try to access fields based on nx API. If this fails, assume only a list of edges was supplied in nx_or_edgelist
    try:
        ids = nx_or_edgelist.nodes
        edges = list(nx_or_edgelist.edges(data=True))
    except AttributeError:
        # allow for user inputted nids to be used if the 'nid' key is set in the optional arg 'nettype'
        ids = kwargs.get('nettype', {}).get('nid', None)
        # otherwise, infer the nodes from the edge list
        if not ids:
            # Need to obtain the set of node ids from the edge list
            edge_arr = np.array(nx_or_edgelist, dtype=object)
            ids = set(edge_arr[:, 0]).union(set(edge_arr[:, 1]))
        # edges will be the unmodified nx_or_edgelist argument
        edges = nx_or_edgelist
    # add the ids of nodes to the network
    G.add_mult(ids=ids, state='S', traced=False)
    # Set the active node list (this is before removing orphans)
    G.node_list = list(G)
    # copy over already generated drawing layout, if one exists
    try:
        G.pos = nx_or_edgelist.pos
    except AttributeError:
        pass
    # normalization factor for edges
    G.norm_factor = K_factor / W_factor if W_factor else 0
    # checking if any there are any edges; agnostic for python iterables and np.arrays
    if is_not_empty(edges):
        # supply edges but do not update counts yet since the first infected have not been set by this point
        G.add_links(edges, update=False)
        # Removing orphans becomes harder to manage at this point
        # if true network needed from here, simply do the degree check, remove appropriately from active node list, and mark the orphans as 'traced'
        # if dual network needed from here, the active nodes are copied over from the true_net
        if rem_orphans:
            for nid in G.nodes:
                if G.degree(nid) == 0:
                    G.node_list.remove(nid)
                    # we mark orphans as isolated (even if NOT technically traced, this will not impact the outcome since orphans have no neighbors)
                    # Note that the node_traced list will be copied over to the tracing network!
                    G.node_traced[nid] = True
    return G


def get_dual_from_predef(G, nx_or_edgelist, count_importance=1, W_factor=0, K_factor=10, **kwargs):
    # the copy of the graph will include everything at this point, including active node_list, node_states, node_traced
    # it will also have a SEPARATE entity for node_counts
    D = G.copy()
    # mark this net as dual
    D.is_dual = True
    # give this network a count_importance -> used to supply different tau rates
    D.count_importance = count_importance
    # used for normalization of weights
    D.norm_factor = K_factor / W_factor if W_factor else 0
    # Try to access fields of nx_or_edgelist based on nx API. If this fails, assume only a list of edges was supplied in nx_or_edgelist
    try:
        edges = list(nx_or_edgelist.edges(data=True))
    except AttributeError:
        edges = nx_or_edgelist
    # clear the true network edges since we want to put the predefined set in
    D.clear_edges()
    # supply edges and norm_factor weight normalization factor
    D.add_links(edges, update=True)
    return D
    
       
def get_dual(G, net_overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, conserve_overlap=True, count_importance=1, nseed=None, active_based_noising=False, **kwargs):
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