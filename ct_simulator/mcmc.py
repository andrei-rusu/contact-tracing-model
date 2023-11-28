import matplotlib.pyplot as plt
import numpy as np
import math as math
import pandas as pd
import theano
import theano.tensor as tt
import pymc3 as pm

from . import run_tracing


def run_loop(state_data, state_population, states=None, use_population=0., sample_kwargs=None):
    """
    Runs a loop of epidemic simulations to perform MCMC sampling of epidemiological parameters using pymc.
    Also generate statistics and trace plots from the MCMC samples.

    Args:
        state_data (pd.DataFrame): DataFrame containing state-level data.
        state_population (dict): Dictionary containing state populations.
        states (list, optional): List of states to loop over. If None, all states in state_data will be used. Default is None.
        use_population (float, optional): Population value to use for sampling. Default is 0.
            If 0, the state population contained in the `state_population` dict will be used.
            If between 0 and 1, the population will be calculated as a fraction of the state population in the `state_population` dict.
            If between -1 and 0, the population will be calculated as a fraction of the last available total value in `state_data`.
            If >= 1, the population will be set to that value.
            If <=-1, the population will be calculated based on the last available total value in `state_data`.
        sample_kwargs (dict, optional): Additional keyword arguments to pass to the `sampleMCMC()` function. Default is None.

    Returns:
        tuple: A tuple containing two DataFrames - trace_results and summary.
            trace_results (pd.DataFrame): DataFrame containing the trace results for each state.
            summary (pd.DataFrame): DataFrame containing the summary statistics for each state.
    """
    if states is None:
        states = state_data['state'].unique()
    if sample_kwargs is None:
        sample_kwargs = {}

    trace_results = []
    summary = []

    for state in states:
        
        data_for_state = state_data.loc[state_data.state == state]
        if use_population == 0:
            pop_for_state = state_population[state]
        elif use_population > 0 and use_population < 1:
            pop_for_state = int(use_population * state_population[state])
        elif use_population >= 1:
            pop_for_state = use_population
        elif use_population <= -1:
            pop_for_state = data_for_state.iloc[-1].total.astype('int32')
        else:
            pop_for_state = int(abs(use_population) * data_for_state.iloc[-1].total)
        print(f'Running {state} with Population {pop_for_state} AND {data_for_state.hospitalized.isna().sum()} hosp NaNs',
              f'AND {data_for_state.death.isna().sum()} death NaNs')
        print("=======================================================================================================")

        try:
            # perform sampling, first set of starting values
            this_sample, this_model = sampleMCMC(data_for_state, pop_for_state, 1, **sample_kwargs)

        except Exception as e:
            print(state + f' failed 1st try with {e}:')
            sample_kwargs['find_map'] = False
            try:
                # perform sampling, second set of starting values
                this_sample, this_model = sampleMCMC(data_for_state, pop_for_state, 1, **sample_kwargs)

            except Exception as e:
                print(state + f' failed 2nd try with {e}:')
                try:
                    # perform sampling, last set of starting values
                    this_sample, this_model = sampleMCMC(data_for_state, pop_for_state, 2, **sample_kwargs)

                except Exception as e:
                    print(state + f' failed 3rd try with {e}:')
                    try:
                        # perform sampling, last set of starting values
                        this_sample, this_model = sampleMCMC(data_for_state, pop_for_state, 3, **sample_kwargs)
                        
                    except Exception as e:
                        trace_results.append({'state':state, 'trace':None, 'model': None})
                        print(state + f' failed 4th try with {e}:')
                        print("=======================================================================================================")
                        continue

        # create summary table
        current_summary = (pm.summary(this_sample, var_names=['i0','beta','gamma','rho','sigma','loss_h','loss_d'], round_to=5, hdi_prob=.95)
            .drop(['mcse_sd','ess_sd','ess_bulk','ess_tail'], axis=1, errors='ignore')
            .reset_index().rename(columns={"index": "param"}))
        current_summary['state'] = state

        # make plots
        pm.plot_trace(this_sample, var_names=('i0','beta','gamma','rho','sigma','loss_h','loss_d'), legend=True, 
                      chain_prop={"color": ['C0', 'C1', 'C2', 'C3', "xkcd:purple blue"]})
        plt.show()

        sum_r_hat = sum(current_summary[1:4]['r_hat'])
        if sum_r_hat > 3.5:
            trace_results.append({"state":state, "trace":None, "model":None})
            print(state + f' failed due to {sum_r_hat=}')
            print("=======================================================================================================")

        else:
            # update summary table
            summary.append(current_summary)
            # update trace table
            trace_results.append({"state":state, "trace":this_sample, "model":this_model})
            print('summary:')
            print("=======================================================================================================")
            print(current_summary)
            print(state + ' succeeded')
            print("=======================================================================================================")
    
    trace_results = pd.DataFrame(trace_results)
    if summary:
        summary = pd.concat(summary, ignore_index=True)
    return trace_results, summary


k_record = 0
@tt.compile.ops.as_op(itypes=[tt.lscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.iscalar],otypes=[tt.iscalar])
def record_vars(i0, beta, gamma, rho_h, rho, interval):
    """
    Records the values of i0, beta, gamma, rho_h, and rho at every `interval`th iteration.

    Args:
        i0 (torch.Tensor): The initial number of infected individuals.
        beta (torch.Tensor): The infection rate.
        gamma (torch.Tensor): The recovery rate.
        rho_h (torch.Tensor): The hospitalization rate.
        rho (torch.Tensor): The mortality rate.
        interval (int): The interval at which to record the variables.

    Returns:
        np.ndarray: The value of i0 as a numpy array.
    """
    i0, beta, gamma, rho_h, rho = i0.item(), beta.item(), gamma.item(), rho_h.item(), rho.item()
    global k_record
    if k_record % interval == 0:
        print(f'Sampled {i0=}; {beta=}; {gamma=}; {rho_h=}; {rho=}')
    k_record += 1
    return np.array(i0)


h_record = 0
d_record = 0
@tt.compile.ops.as_op(itypes=[tt.dvector, tt.dvector, tt.dscalar, tt.iscalar],otypes=[tt.dscalar])
def get_loss(x1, x2, limit, q):
    """
    Calculates the mean squared error between two arrays, x1 and x2, and returns the result.
    If q is 0, the error is compared to the global variable h_record and printed if it exceeds the limit.
    If q is not 0, the error is compared to the global variable d_record and printed if it exceeds the limit.
    
    Args:
    x1 (numpy.ndarray): The first array to compare.
    x2 (numpy.ndarray): The second array to compare.
    limit (float): The limit for the error difference to trigger a print statement.
    q (int): A flag to determine which global variable to compare the error to.
    
    Returns:
    float: The mean squared error between x1 and x2.
    """
    err = ((x1 - x2) ** 2).mean()
    if q == 0:
        global h_record
        if abs(h_record - err) >= limit:
            print(f'Hosp error: {err}')
        h_record = err
    else:
        global d_record
        if abs(d_record - err) >= limit:
            print(f'Death error: {err}')
        d_record = err
    return err


@tt.compile.ops.as_op(itypes=[tt.lscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.iscalar, tt.iscalar, tt.iscalar, tt.iscalar],otypes=[tt.dvector, tt.dvector])
def call_seirt(i0, beta, gamma, rho_h, rho, pop, length, model, print_try):
    """
    Runs a mock simulation of an epidemic using the SEIR-T or SIR-T model and returns the total number of recovered and/or
    hospitalized individuals and deaths.

    Args:
        i0 (float): The initial number of infected individuals.
        beta (float): The transmission rate.
        gamma (float): The recovery rate.
        rho_h (float): The hospitalization rate.
        rho (float): The death rate.
        pop (float): The population size.
        length (float): The length of the simulation.
        model (int): The epidemic model to use (0=SIR, 1=SEIR, 2=COVID).
        print_try (bool): Whether to print the simulation parameters and results.

    Returns:
        tuple: A tuple containing the total number of recovered and/or hospitalized individuals and deaths.
    """
    i0, beta, gamma, rho_h, rho, pop, length, model, print_try = \
        i0.item(), beta.item(), gamma.item(), rho_h.item(), rho.item(), pop.item(), length.item(), model.item(), print_try.item()
    epidemic_models = ['sir', 'seir', 'covid']
    if print_try:
        print(f' - Trying with model={epidemic_models[model]}; {i0=}; {beta=}; {gamma=}; {rho_h=}; {rho=} ...')
    # in our model, rho controls H->D instead of I/R->D, therefore rho needs scaling (iff rho_h != 0)
    if rho_h != 0:
        rho /= rho_h
    st, _ = run_tracing.run_api(
                    first_inf=i0, presample=10000, pa=.2, taur=0, taut=0, beta=beta, gamma=gamma, ph=rho_h, lamdahd=rho,
                    nettype='barabasi', netsize=pop, k=5, p=.01, use_weights=True, dual=0,
                    seed=21, netseed=25, infseed=-1, model=epidemic_models[model], spontan=True, sampling_type='min',
                    multip=0, nnets=1, niters=1, nevents=length, sim_agent_based=1,
                    animate=-2, summary_print=0, summary_splits=length, avg_without_earlystop=True)
    if print_try:
        print(' - Results: Inf ->', st['res']['average-overall-infected']['mean'], 
              '  Rec ->', st['res']['average-overall-recovered']['mean'], 
              '  Hos ->', st['res']['average-overall-hospital']['mean'],
              '  Ded ->', st['res']['average-overall-death']['mean'])
    if model < 2:
        return np.array([int(r['mean']) for r in st['res']['average-total-recovered']], dtype=float), np.zeros(1, dtype=float)
    else:
        return np.array([int(r['mean']) for r in st['res']['average-total-hospital']], dtype=float), np.array([int(r['mean']) for r in st['res']['average-total-death']], dtype=float)
    
    
def sampleMCMC(data, pop=1000, start_val=1, ode=True, i0=None, beta=None, gamma=0, rho_h=0, rho=0, model=0., fit_stats=('d',), 
               find_map=False, print_try=0, print_interval=5, print_err_limit=.05, step=None, **kwargs):
    """
    Runs a Markov Chain Monte Carlo (MCMC) simulation to fit a compartmental model to COVID-19 data.

    Args:
        data (pandas.DataFrame): A DataFrame containing COVID-19 data.
        pop (int): The population size.
        start_val (int): An integer indicating the starting point for the simulation.
            If 1, the starting value for x will either be x_start, if provided, or (x_recorded/pop)/2, otherwise.
            If 2, the starting value for x will be (x_recorded/total + x_recorded/pop)/2
            Otherwise, a default value will be used for each.
        ode (bool): A boolean indicating whether to use an ODE model or IBMF model (default True).
        i0 (float): The initial number of infected individuals (default None).
        beta (float): The starting value for the beta parameter (default None).
        gamma (float): The starting value for the gamma parameter (default 0).
        rho_h (float): The starting value for the rho_h parameter (default 0).
        rho (float): The starting value for the rho parameter (default 0).
        model (int): The epidemic model to use (0=SIR, 1=SEIR, 2=COVID).
        fit_stats (tuple): A tuple containing the statistics to fit (default ('d',)).
        find_map (bool): A boolean indicating whether to find the maximum a posteriori (MAP) estimate (default False).
        print_try (int): The number of times to try to print the progress (default 0).
        print_interval (int): The interval at which to print the progress (default 5).
        print_err_limit (float): The error limit for printing progress (default 0.05).
        step (dict): A dictionary containing the step method and its arguments (default None).
        **kwargs: Additional keyword arguments to pass to the `pm.sample()` function.

    Returns:
        None
    """
    fit_hosp = 'h' in fit_stats
    fit_death = 'd' in fit_stats
    length = len(data)
    # splitting data into infections and time as numpy arrays
    data_hospital = data['hospitalized'].to_numpy()
    data_death = data['death'].to_numpy()
    time = np.linspace(0, length-1, length)
    global k_record, h_record, d_record
    k_record = h_record = d_record = 0
    
    # establishing model
    with pm.Model() as pm_model:
        
        # create population number priors at the beginning of the simulation
        i0 = pm.Poisson('i0', mu=i0 if i0 else pop/1000)
        s0 = pm.Deterministic('s0', pop - i0)
        
        # extract starting components
        pos = float(data['positive'].iloc[-1])
        rec = float(data['recovered'].iloc[-1])
        hos = float(data['hospitalized'].iloc[-1])
        dea = float(data['death'].iloc[-1])
        tot = float(data['total'].iloc[-1])
        
        # create starting values based on data, does not inform inference but starts at a reasonable value
        # start_val beta conditional on start_val argument
        beta = \
            (((pos/tot)/2 if not beta else beta) if start_val==1 else
            (pos/tot + pos/pop)/2 if start_val == 2 else .05)
        
        # start_val gamma conditional on start_val argument
        gamma = \
            (((rec/tot)/2 if not gamma else gamma) if start_val==1 else
            (rec/tot + rec/pop)/2 if start_val==2 else .047)
        
        # start_val rho conditional on start_val argument
        rho_h = \
            (((hos/pos)/2 if not rho_h else rho_h) if start_val==1 else
            (hos/pos + hos/pop)/2 if start_val==2 else .036)
        
        # start_val rho conditional on start_val argument
        rho = \
            (((dea/pos)/2 if not rho else rho) if start_val==1 else
            (dea/pos + dea/pop)/2 if start_val==2 else .036)
        
        print(f'{pos=}, {rec=}, {hos=}, {dea=}, {tot=}, {beta=:.5f}, {gamma=:.5f}, {rho_h=:.5f}, {rho=:.5f}')
        
        # creating priors for beta, gamma, and rho
        beta = pm.InverseGamma('beta', mu=.05, sigma=.5, testval=beta)
        gamma = pm.InverseGamma('gamma', mu=.047, sigma=.5, testval=gamma)
        rho_h = pm.TruncatedNormal('rho_h', mu=.056, sigma=.01, lower=0, upper=1, testval=rho_h) \
                    if fit_hosp else theano.shared(rho_h)
        rho = pm.TruncatedNormal('rho', mu=.036, sigma=.01, lower=0, upper=1, testval=rho) \
                    if fit_death else theano.shared(rho)
        # create variance prior
        sigma = pm.HalfCauchy('sigma', beta=4)

        if ode:
            # create number of removed based on analytic solution and above parameters
            sir_rem = pm.Deterministic('sir_rem',
                pop - ((s0 + i0)**(beta/(beta - gamma)))*
                (s0 + i0*tt.exp(time*(beta - gamma)))**(-gamma/(beta - gamma)))
            
            if fit_hosp:
                sir_hospital = pm.Deterministic('sir_hospital', rho_h*sir_rem)
                obs_hospital = pm.TruncatedNormal('obs_hospital', mu=sir_hospital, sigma=sigma,
                                                     lower=0, upper=pop, observed=data_hospital)
            if fit_death:
                sir_death = pm.Deterministic('sir_death', rho*sir_rem)
                obs_death = pm.TruncatedNormal('obs_death', mu=sir_death, sigma=sigma,
                                                     lower=0, upper=pop, observed=data_death)
            if step:
                for entry, entry_kwargs in step.items():
                    step = getattr(pm, entry)(**entry_kwargs)
            else:
                step=pm.NUTS(target_accept=.99)
        else:
            if model < 2:
                rem, _ = call_seirt(i0, beta, gamma, theano.shared(0.), theano.shared(0.),
                                    theano.shared(pop), theano.shared(length), theano.shared(model), theano.shared(print_try))
                no_grad = (gamma, beta)
                w_grad = [sigma]
                sir_rem = pm.Deterministic('sir_rem', rem)
                if fit_hosp:
                    sir_hospital = pm.Deterministic('sir_hospital', rho_h*sir_rem)
                    obs_hospital = pm.TruncatedNormal('obs_hospital', mu=sir_hospital, sigma=sigma,
                                                         lower=0, upper=pop, observed=data_hospital)
                    w_grad.insert(0, rho_h)
                if fit_death:
                    sir_death = pm.Deterministic('sir_death', rho*sir_rem)
                    obs_death = pm.TruncatedNormal('obs_death', mu=sir_death, sigma=sigma,
                                                         lower=0, upper=pop, observed=data_death)
                    w_grad.insert(0, rho)
            else:
                rem, ded = call_seirt(i0, beta, gamma, rho_h, rho, theano.shared(pop), theano.shared(length), theano.shared(model), theano.shared(print_try))
                no_grad = [gamma, beta]
                w_grad = (sigma,)
                if fit_hosp:
                    sir_hospital = pm.Deterministic('sir_hospital', rem)
                    obs_hospital = pm.TruncatedNormal('obs_hospital', mu=sir_hospital, sigma=sigma,
                                                         lower=0, upper=pop, observed=data_hospital)
                    no_grad.insert(0, rho_h)
                if fit_death:
                    sir_death = pm.Deterministic('sir_death', ded)
                    obs_death = pm.TruncatedNormal('obs_death', mu=sir_death, sigma=sigma,
                                                         lower=0, upper=pop, observed=data_death)
                    no_grad.insert(0, rho)
            if step:
                for entry, entry_kwargs in step.items():
                    step = getattr(pm, entry)(**entry_kwargs)
            else:
                # step can be replaced by `None` here to utilize the default behavior of pm.sample
                step=[pm.Metropolis(vars=(i0)), pm.Slice(vars=no_grad), pm.NUTS(vars=w_grad, target_accept=.99)]
                
        pm.Deterministic('vars', record_vars(i0, beta, gamma, rho_h, rho, theano.shared(print_interval)))
        if fit_hosp:
            pm.Deterministic('loss_h', get_loss(sir_hospital, obs_hospital, theano.shared(print_err_limit), theano.shared(0)))
        else:
            pm.Deterministic('loss_h', theano.shared(0))
        if fit_death:
            pm.Deterministic('loss_d', get_loss(sir_death, obs_death, theano.shared(print_err_limit), theano.shared(1)))
        else:
            pm.Deterministic('loss_d', theano.shared(0))

        # specifying model conditions
        maxval = 1000 if ode else 100
        start=pm.find_MAP(maxeval=maxval) if find_map else None
        # execute sampling
        model_trace = pm.sample(start=start, step=step, return_inferencedata=True, **kwargs)

    # return posterior samples and other information
    return model_trace, pm_model