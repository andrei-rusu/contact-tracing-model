import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid


pa_vals = [.2]
tau_vals = np.array([.05, .1, .2, .5])
GRID = [
    # When taut = 0, it does not matter the degree of overlap or uptake (so make them 1)
    {'uptake': [1],
     'taut': [0],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1],
    },
    # When taut != 0, try all overlaps or uptakes
    {'uptake': np.linspace(.1, 1, 10),
     'taut': tau_vals,
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1] # [.05, .1, .3, .5, 1],
    }
]
sampler = ParameterGrid(GRID)


def len_grid():
    return len(sampler)
    
def get_job_ids():
    print('0-' + str(len_grid() - 1))

def get_parameters_for_id(jid):
    # the order of variables is given by string name in decreasing order
    uptake, taut, taur, pa, overlap = sampler[jid].values()
    roundprint(uptake, taut, taur, pa, overlap)
    
def roundprint(*args, **kwargs):
    for arg in args:
        print(round(arg, 2), end=' ', **kwargs)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--id', type=int, default=0)
    args = argparser.parse_args()
    get_parameters_for_id(args.id)