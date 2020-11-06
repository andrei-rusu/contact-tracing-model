import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid



pa_vals = [.2, .5]
tau_vals = np.array([0, .03, .05, .1, .2, .5])
GRID = [
    # When taut = 0, it does not matter the degree of overlap (so only try overlap=1)
    {'taut': [0],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1],
    },
    # When taut != 0, try all overlaps
    {'taut': tau_vals[1:],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [.05, .1, .3, .5, 1],
    }
]
sampler = ParameterGrid(GRID)


def get_len_grid():
    print(len(sampler))

def get_parameters_for_id(jid):
    # the order of variables is given by string name in decreasing order
    taut, taur, pa, overlap = sampler[jid].values()
    print(taut, taur, pa, overlap)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--id', type=int, default=0)
    args = argparser.parse_args()
    get_parameters_for_id(args.id)