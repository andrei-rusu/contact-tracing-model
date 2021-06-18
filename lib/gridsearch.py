import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid


pa_vals = [.2]
tau_vals = np.array([.1, .2, .5, 1., 1.5, 2.])
GRID = [
    # When taut = 0, it does not matter whether its dual/triad, or the degree of overlap/uptake (so make them 1)
#     {'uptake': [1],
#      'taut': [0],
#      'taur': tau_vals,
#      'pa': pa_vals,
#      'overlap': [1],
#      'dual': [1],
#     },
    # When taut != 0 and dual=1 scenario, try all overlaps OR uptakes (used for digital tracing only)
    {'uptake': np.linspace(.1, 1., 7),
     'taut': [10],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1], # [.05, .1, .3, .5, 1],
     'dual': [1],
    },
    # When taut != 0 and dual=2 (triad) scenario, all uptakes will be used for the first net, and overlaps for second net
    {'uptake': np.linspace(.1, 1., 7),
     'taut': [10],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': np.linspace(.1, 1., 7), # [.05, .1, .3, .5, 1],
     'dual': [2],
    }
]
sampler = ParameterGrid(GRID)


def len_grid():
    return len(sampler)
   
    
def get_job_ids():
    print('0-' + str(len_grid() - 1))

    
def get_parameters_for_id(jid):
    # the order of variables is given by string name in decreasing order
    uptake, taut, taur, pa, overlap, dual = sampler[jid].values()
    roundprint(uptake, taut, taur, pa, overlap, dual)
    
    
def roundprint(*args, **kwargs):
    for arg in args:
        print(round(arg, 2), end=' ', **kwargs)
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--id', type=int, default=0)
    # parse_known_args is needed for the script to be runnable from Notebooks
    args, unknown = argparser.parse_known_args()
    get_parameters_for_id(args.id)