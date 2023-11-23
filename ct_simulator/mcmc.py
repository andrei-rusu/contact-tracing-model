import matplotlib.pyplot as plt
import numpy as np
import math as math

import theano
import theano.tensor as tt
import pymc3 as pm

from . import run_tracing


def plot_cont(self, ax=None, size=1000, bounds=None, log=False, **kwargs):
    """
    Plots the continuous distribution of the MCMC samples if assigned to pm.Continous.

    Args:
        ax (matplotlib.axes.Axes, optional): The axes on which to plot the distribution. If not provided, a new figure and axes will be created.
        size (int, optional): The number of samples to generate if bounds are not provided. Defaults to 1000.
        bounds (tuple, optional): The bounds of the distribution. If not provided, the bounds will be inferred from the generated samples.
        log (bool, optional): Whether to plot the log probabilities or the true probabilities. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        matplotlib.axes.Axes: The axes on which the distribution was plotted.
    """
    if ax is None:
        _, ax = plt.subplots()
    if bounds:
        mn, mx = bounds
    else:
        samples = self.random(size=size)
        mn, mx = np.min(samples), np.max(samples)
    x = np.linspace(mn, mx, size)
    p = self.logp(x)
    ax.plot(x, (p if log else np.exp(p)).eval(), **kwargs)
    return ax

# Assign plotting function to the abstract class pm.Continuous
pm.Continuous.plot = plot_cont


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
    # in our model, rho controls H->D instead of I/R->D, therefore the rate needs scaling
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
    
    
def sampleMCMC(data, pop, start, ode=True, i0=None, beta_start=None, gamma_start=0, rho_h_start=0, rho_start=0, model=0., fit_stats=('d',), find_map=False, print_try=0, print_interval=5, print_err_limit=.05, step=None, **kwargs):
    """
    Runs a Markov Chain Monte Carlo (MCMC) simulation to fit a compartmental model to COVID-19 data.

    Args:
        data (pandas.DataFrame): A DataFrame containing COVID-19 data.
        pop (int): The population size.
        start (int): An integer indicating the starting point for the simulation. 1 for the last data point, 2 for the average of the last 7 days, and 3 for a fixed value.
        ode (bool): A boolean indicating whether to use an ODE model or IBMF model (default True).
        i0 (float): The initial number of infected individuals (default None).
        beta_start (float): The starting value for the beta parameter (default None).
        gamma_start (float): The starting value for the gamma parameter (default 0).
        rho_h_start (float): The starting value for the rho_h parameter (default 0).
        rho_start (float): The starting value for the rho parameter (default 0).
        model (int): The epidemic model to use (0=SIR, 1=SEIR, 2=COVID).
        fit_stats (tuple): A tuple containing the statistics to fit (default ('d',)).
        find_map (bool): A boolean indicating whether to find the maximum a posteriori (MAP) estimate (default False).
        print_try (int): The number of times to try to print the progress (default 0).
        print_interval (int): The interval at which to print the progress (default 5).
        print_err_limit (float): The error limit for printing progress (default 0.05).
        step (dict): A dictionary containing the step method and its arguments (default None).
        **kwargs: Additional keyword arguments.

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
    with pm.Model() as model:
        
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
        # start beta conditional on start argument
        beta_start = \
            (((pos/tot)/2 if not beta_start else beta_start) if start==1 else
            (pos/tot + pos/pop)/2 if start == 2 else .05)
        
        # start gamma conditional on start argument
        gamma_start = \
            (((rec/tot)/2 if not gamma_start else gamma_start) if start==1 else
            (rec/tot + rec/pop)/2 if start==2 else .047)
        
        # start rho conditional on start argument
        rho_h_start = \
            (((hos/pos)/2 if not rho_h_start else rho_h_start) if start==1 else
            (hos/pos + hos/pop)/2 if start==2 else .036)
        
        # start rho conditional on start argument
        rho_start = \
            (((dea/pos)/2 if not rho_start else rho_start) if start==1 else
            (dea/pos + dea/pop)/2 if start==2 else .036)
        
        print(f'{pos=}, {rec=}, {hos=}, {dea=}, {tot=}, {beta_start=:.5f}, {gamma_start=:.5f}, {rho_h_start=:.5f}, {rho_start=:.5f}')
        
        # creating priors for beta, gamma, and rho
        beta = pm.InverseGamma('beta', mu=.05, sigma=.5, testval=beta_start)
        gamma = pm.InverseGamma('gamma', mu=.047, sigma=.5, testval=gamma_start)
        rho_h = pm.TruncatedNormal('rho_h', mu=.056, sigma=.01, lower=0, upper=1, testval=rho_h_start) \
                    if fit_hosp else theano.shared(rho_h_start)
        rho = pm.TruncatedNormal('rho', mu=.036, sigma=.01, lower=0, upper=1, testval=rho_start) \
                    if fit_death else theano.shared(rho_start)
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
    return model_trace, model