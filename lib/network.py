import networkx as nx

import itertools
import random
from copy import deepcopy
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lib.utils import rand_pairs, rand_pairs_excluding, get_z_for_overlap, get_overlap_for_z

STATES_COLOR_MAP = {
    'S': 'darkgreen',
    'E': 'yellow',
    'I': 'orange',
    'Ia': 'pink',
    'Is': 'red',
    'H': 'cyan',
    'T': 'blue',
    'R': 'lime',
    'D': 'gray'
}

class Network(nx.Graph):
    
    def __init__(self, **kwds):
        # Atomic Counter for node ids
        self.cont = itertools.count()
        # Initialize pos to None - a new spring layout is created when first drawn
        self.pos = None
        # Overlap after noising - initially perfect overlap, changes after calling 'noising_links'
        self.overlap = 1
        # Maintain node states and neighbor count
        self.node_list = []
        self.node_states = []
        self.node_traced = []
        self.node_counts = defaultdict(dict)
        
        super().__init__(**kwds)
        
    def init_random(self, n=200, k=10, weighted=False):
        # Add nodes in state 'S'
        self.add_mult(n, 'S')
        # Add random links with degree k
        remaining_links = int(n * k / 2)
        # Generate random pairs without replacement
        links_to_create = rand_pairs(n, remaining_links)
        if weighted:
            # Create random weights from 1 to 10 and append them to the list of edge tuples
            weights = random.choices(range(1,10), k=remaining_links)
            links_to_create = [links_to_create[i] + (weights[i],) for i in range(remaining_links)]
        # Add the random edges with/without weights depending on 'weighted' parameter
        self.add_links(links_to_create)
                
        return self
        
    def noising_links(self, overlap=None, z_add=0, z_rem=5, weighted=False, update=True):
        # Recover average degree
        k = self.avg_degree()
        # Recover num of nodes
        n = len(self.node_list)
        
        # If no overlap value, use z_add & z_rem
        if overlap is None:
            # Average degree k, z_add, z_rem used to calculate overlap after noising
            self.overlap = get_overlap_for_z(k, z_add, z_rem)
        # If overlap given, calculate either z_add and z_rem
        else:
            self.overlap = overlap
            # if z_add = 0, z_add will stay 0 while z_rem = z; OTHERWISE z_add = z_rem = z
            z_add, z_rem = get_z_for_overlap(k, overlap, include_add = z_add)
            
        # Current edge set (prior to noising, nodes in each edge tuple are sorted)
        current_edges = {tuple(sorted(edge)) for edge in self.edges}
                        
        # Random adding z_add on average - this may be 0
        remaining_links_add = int(n * z_add / 2)
        links_to_add = rand_pairs_excluding(n, remaining_links_add, to_exclude=current_edges) if z_add else []
                    
        # Random removing z_rem on average
        remaining_links_rem = int(n * z_rem / 2)
        links_to_rem = random.sample(current_edges, remaining_links_rem)
        
        self.add_edges_from(links_to_add)
        self.remove_edges_from(links_to_rem)
        
        if update:
            self.update_counts()
        
        return self
                
    
    def init_states(self, state='S'):
        # Set all nodes back to S
        self.node_states = [state] * len(self.node_list)
 

    def add(self, state='S'):
        current = next(self.cont)
        self.add_node(current)
        # Update cached node list and states
        self.node_list.append(current)
        self.node_states.append(state)
        self.node_traced.append(False)
        
    def add_mult(self, n=200, state='S'):
        current = next(self.cont)
        self.add_nodes_from(range(current, current + n))
        # Update cached node list and states
        self.node_list = list(self)
        self.node_states += [state] * n
        self.node_traced += [False] * n
        # increment count to reflect n additions
        self.cont = itertools.count(current + n)
        
    def update_counts(self, nlist=None):
        # if nlist not given, update_counts for all nodes
        if nlist is None:
            nlist = self.node_list
        for nid in nlist:
            # get the list of neighbour states and num of traced
            neigh_states_for_node = [self.node_states[i] for i in self.neighbors(nid)]
            # update counts
            self.node_counts[nid] = Counter(neigh_states_for_node)
            
    def update_counts_with_traced(self, nlist=None):
        # if nlist not given, update_counts for all nodes
        if nlist is None:
            nlist = self.node_list
        for nid in nlist:
            # get the list of neighbour states and num of traced
            neigh_states_for_node = []
            num_traced_for_node = 0
            for i in self.neighbors(nid):
                neigh_states_for_node.append(self.node_states[i])
                if self.node_traced[i]:
                    num_traced_for_node += 1
            # update neighbor count but give priority to self.node_traced for 'T' count
            counter = Counter(neigh_states_for_node)
            counter['T'] = num_traced_for_node
            self.node_counts[nid] = counter
            
    def get_count(self, nid, state='I'):
        return self.node_counts[nid][state]
            
    def generate_layout(self, seed=43):
        self.pos = nx.spring_layout(self, seed=seed)
        
    def get_state(self, nid):
        return self.node_states[nid]
        
    def change_state(self, nid, state='I', update=True):
        self.node_states[nid] = state
        if update:
            self.update_counts()
                
    def change_state_fast_update(self, nid, state):
        old_state = self.node_states[nid]
        # update counts if NOT traced, NOT hospitalized, NOT exposed without being infectious:
        if state != 'E' and not self.node_traced[nid] and old_state != 'H':
            old_state = self.node_states[nid]
            for neigh in self.neighbors(nid):
                self.node_counts[neigh][old_state] -= 1
                self.node_counts[neigh][state] += 1
        self.node_states[nid] = state
            
    def change_traced_state_fast_update(self, nid):
        """
        When using separate_traced, the T state is independent of all the other states and represented via a flag
        Counts are updated as-if this would be an actual state change: 
         - node_counts for neighbor node -> for current state of nid decrement, for 'T' increment
        """
        self.node_traced[nid] = True
        inf_state = self.node_states[nid]
        for neigh in self.neighbors(nid):
            self.node_counts[neigh][inf_state] -= 1
            self.node_counts[neigh]['T'] += 1
                
    def add_link(self, nid1, nid2, weight=1, update=False):
        self.add_edge(nid1, nid2, weight=weight)
        if update:
            self.update_counts([nid1, nid2])
            
    def rem_link(self, nid1, nid2, update=False):
        self.remove_edge(nid1, nid2)
        if update:
            self.update_counts([nid1, nid2])
            
    def add_links(self, lst, weight=1, update=False):
        len_elem = len(lst[0])
        # If only source + target nodes provided, add specified 'weight' parameter as weight
        if len_elem == 2:
            self.add_edges_from(lst, weight=weight)
        # If weights provided alognside nodes, ignore weight parameter
        elif len_elem == 3:
            self.add_weighted_edges_from(lst)
        if update:
            self.update_counts()
            
    def avg_degree(self):
        return np.mean(list(dict(self.degree()).values()))
        
    def draw(self, pos=None, show=True, seed=43):
        # for the true network, colors for all nodes are based on their state
        if self.overlap == 1:
            colors = list(map(lambda x: STATES_COLOR_MAP[x], self.node_states))
        # for the dual network, when no explicit T state set, give priority to self.node_traced == True as if 'T' state
        else:
            colors = [STATES_COLOR_MAP['T'] if traced else STATES_COLOR_MAP[state] for state, traced in zip(self.node_states, self.node_traced)]
        # by doing this we avoid overwriting self.pos with pos when we just want a different layout for the drawing
        pos = pos if pos else self.pos
        # if both were None, generate a new layout with the seed and use it for drawing
        if not pos:
            self.generate_layout(seed)
            pos = self.pos
        # draw graph
        nx.draw(self, pos=pos, node_color=colors, with_labels=True)
        # create color legend
        plt.legend(handles=[mpatches.Patch(color=color, label=state) for state, color in STATES_COLOR_MAP.items()])
        
        if show:
            plt.show()    
    
def get_random(n=200, k=10, weighted=False):
    G = Network()
    return G.init_random(n, k, weighted)
       
def get_dual(G, overlap=None, z_add=0, z_rem=5, weighted=False):
    N = deepcopy(G)
    return N.noising_links(overlap, z_add, z_rem, weighted, update=True)