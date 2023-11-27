AGENT_SOURCE_DEFAULT = 'control_diffusion'


def create_agent(source=AGENT_SOURCE_DEFAULT, **agent_params):
    """ 
    Creates and returns a testing, tracing or vaccination control agent object using the specified parameters.

    Args:
        source (str): The source of the agent implementation, e.g. 'control_diffusion', etc.
            'local' will import the agent module from the main package directory.
        agent_params (dict): The parameters for the agent, e.g. {'typ': 'centrality', 'measure': 'degree'}, etc.
    Returns:
        Agent: An agent object that implements the control() method for ranking nodes, and optionally a finish() method.
    """
    # Import the agent module from the specified source
    if source == 'local':
        try:
            from agent import Agent
        except ImportError:
            raise ImportError('Could not import an agent implementation from local source. Please check that one exists in the main package directory.')
        return Agent.from_dict(**agent_params)
    elif source == 'control_diffusion':
        from control_diffusion import Agent
        return Agent.from_dict(**agent_params)
    else:
        raise ValueError(f'Unrecognized agent source: {source}')
    

def create_model(source=AGENT_SOURCE_DEFAULT, **model_params):
    """ 
    Creates and returns a GNN model using the specified parameters.

    Args:
        source (str): The source of the agent implementation, e.g. 'control_diffusion', etc.
            'local' will import the agent module from the main package directory.
        model_params (dict): The parameters for the model, e.g. {'lr': 0.01, 'rl_sampler': None}, etc.
    Returns:
        Model: A GNN model that can rank nodes.
    """
    # Import the agent module from the specified source
    if source == 'local':
        try:
            from agent import Agent
        except ImportError:
            raise ImportError('Could not import an agent implementation from local source. Please check that one exists in the main package directory.')
        return Agent.model_from_dict(**model_params)
    elif source == 'control_diffusion':
        from control_diffusion import Agent
        return Agent.model_from_dict(**model_params)
    else:
        raise ValueError(f'Unrecognized agent source: {source}')