import os
import argparse
import json
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler


AGENT_CONFIG_PATH = 'config/agent_config.json'
AMODEL_CONFIG_PATH = 'config/amodel_config.json'
NEW_AGENT_CONFIG_FOLDER = 'temp/'
NEW_AMODEL_CONFIG_FOLDER = 'temp/'

pa_vals = [.2, .5]
tau_vals = [.1, .2]
group_vals = [-1, 1, 2]
GRID = [
    # # When taut = 0, it does not matter whether its dual/triad, or the degree of overlap/uptake (so make them 1)
    {'uptake': [1],
     'taut': [0],
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1],
     'group': group_vals,
     'dual': [1],
    },
    # When taut != 0 and dual=1 scenario, try all overlaps OR uptakes (used for digital tracing only)
    {'uptake': np.linspace(.1, 1., 4),
     'taut': tau_vals,
     'taur': tau_vals,
     'pa': pa_vals,
     'overlap': [1],
     'group': group_vals,
     'dual': [1],
    },
    # # When taut != 0 and dual=2 (triad) scenario, all uptakes will be used for the first net, and overlaps for second net
    # {'uptake': np.linspace(.1, 1., 7),
    #  'taut': [10],
    #  'taur': tau_vals,
    #  'pa': pa_vals,
    #  'overlap': np.linspace(.1, 1., 7), # [.05, .1, .3, .5, 1],
    #  'group': group_vals,
    #  'dual': [2],
    # }
]

AGENT_GRID = [
    {'m:layer_name': ['GAT', 'GIN'],
     'a:typ': ['sl'],
     'a:rl_sampler': ['', 'softmax'],
     'control_initial_known': [0.25, 0.5],
     'control_after': [3, 5]
    },
]


sampler = None
def create_sampler(sample=0, state=0, agent_file=0):
    """
    Create a sampler object for hyperparameter tuning.

    Args:
        sample (int): Number of samples to generate from the parameter grid. If 0, returns the entire grid.
        state (int): Random seed for the sampler.
        agent_file (int): Whether to use GRID or AGENT_GRID for sampling. Defaults to 0 (GRID).

    Returns:
        list or ParameterGrid: A list of parameter samples if `sample` is non-zero, otherwise the entire parameter grid.
    """
    global sampler 
    grid = AGENT_GRID if agent_file else GRID
    sampler = list(ParameterSampler(grid, n_iter=sample, random_state=state)) if sample else ParameterGrid(grid)
    return sampler


def len_sampler():
    """
    Returns the length of the sampler object.

    Returns:
        int: The length of the sampler object.
    """
    return len(sampler)


def get_job_ids():
    """
    Returns a string representing the range of job IDs for the current sampler.

    Returns:
        str: A string representing the range of job IDs for the current sampler.
    """
    return f'0-{len(sampler) - 1}'


def std_write(*args, **kwargs):
    """
    Prints the given arguments to the console, rounding any floats to 2 decimal places.
    This will effectively make them available to a bash script as variables.

    Args:
        *args: The arguments to print.
        **kwargs: Additional keyword arguments to pass to the print function.
    """
    for arg in args:
        print(round(arg, 2) if isinstance(arg, float) else arg, end=' ', **kwargs)


def get_parameters_for_id(job_id, agent_file=0):
    """
    Writes the parameters for a given job ID using `std_write`.

    Args:
        job_id (int): The ID of the job to get parameters for.
        agent_file (int, optional): Whether to replace the agent and model parameters from the config JSON files with the
            values present in the parameter sampler, effectively enabling parameter search for these. Defaults to 0.

    Returns:
        int: The number of epidemic parameters returned.

    Notes:
        The epidemic simulator parameters are always printed out, regardless of the value of `agent_file`.
        The original config files are not updated in-place, but new ones are created in the NEW_AGENT_CONFIG_FOLDER folder.
    """
    assert job_id is not None, 'job_id must be specified'
    simulator_params = reconcile_agent_json(job_id) if agent_file else sampler[job_id]
    std_write(*simulator_params)
    return len(simulator_params)


def reconcile_agent_json(job_id=None, agent_params=None, amodel_params=None):
    """
    Creates new `agent` and `amodel` config files with the 'a:' and 'm:' parameters from the parameter sampler.
    Also filters out the 'a:' and 'm:' entries from the sampler, effectively returning only the epidemic simulator parameters.

    Args:
        job_id (int): The ID of the job to get parameters for.
        agent_params (dict, optional): Additional agent parameters to update.
        amodel_params (dict, optional): Additional amodel parameters to update.
    """
    with open(AGENT_CONFIG_PATH, 'r', encoding='utf8') as f:
        agent = json.loads(f.read())
    with open(AMODEL_CONFIG_PATH, 'r', encoding='utf8') as f:
        amodel = json.loads(f.read())
    simulator_params = []
    agent_modif = amodel_modif = False
    if job_id is not None:
        params = sampler[job_id]
        for k, v in params.items():
            if k.__contains__('a:'):
                agent[k[2:]] = v
                agent_modif = True
            elif k.__contains__('m:'):
                amodel[k[2:]] = v
                amodel_modif = True
            else:
                simulator_params.append(v)
    if agent_params:
        agent.update(agent_params)
        agent_modif = True
    if amodel_params:
        amodel.update(amodel_params)
        amodel_modif = True
    if agent_modif:
        print('New agent config created')
        if not os.path.exists(NEW_AGENT_CONFIG_FOLDER):
            os.makedirs(NEW_AGENT_CONFIG_FOLDER)
        with open(f'{NEW_AGENT_CONFIG_FOLDER}agent_{job_id}.json', 'r', encoding='utf8') as f:
            json.dump(agent, f)
    if amodel_modif:
        print('New model config created')
        if not os.path.exists(NEW_AMODEL_CONFIG_FOLDER):
            os.makedirs(NEW_AMODEL_CONFIG_FOLDER)
        with open(f'{NEW_AMODEL_CONFIG_FOLDER}amodel_{job_id}.json', 'r', encoding='utf8') as f:
            json.dump(amodel, f)
    return simulator_params


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sample', type=int, default=0)
    argparser.add_argument('--state', type=int, default=0)
    argparser.add_argument('--agent_file', type=int, default=0)
    argparser.add_argument('--id', type=int, default=0)
    # parse_known_args is needed for the script to be runnable from Notebooks
    args, _ = argparser.parse_known_args()
    create_sampler(args.sample, args.state, args.agent_file)
    get_parameters_for_id(args.id, args.agent_file)