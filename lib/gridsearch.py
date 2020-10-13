import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid

GRID = [
    # When taut = 0, it does not matter the degree of overlap (so only try overlap=1)
    {'taut': [0],
     'taur': np.append(np.array([.001, .005]), np.linspace(.01, .1, 4)),
     'overlap': [1],
    },
    {'taut': np.append(np.array([.001, .005]), np.linspace(.01, .1, 4)),
     'taur': np.append(np.array([.001, .005]), np.linspace(.01, .1, 4)),
     'overlap': np.linspace(0, 1, 10),
    }
]

def get_parameters_for_id(jid):
    sampler = ParameterGrid(GRID)
    taut, taur, overlap = sampler[jid].values()
    print(taut, taur, overlap)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--id', type=int, default=0)
    args = argparser.parse_args()
    get_parameters_for_id(args.id)