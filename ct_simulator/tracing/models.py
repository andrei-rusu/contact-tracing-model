from collections import defaultdict
from .utils import get_stateless_sampling_func, get_stateful_sampling_func


def get_transitions_for_model(args, exp):
    """
    Returns two defaultdict objects containing the transition parameters for the true_net and know_net models.

    Args:
    args (argparse.Namespace): An object containing the command-line arguments passed to the script.
    exp (bool): A boolean indicating whether the rates need to be exponentiated or not.

    Returns:
    trans_true (defaultdict): A defaultdict object containing the transition parameters for the true_net model.
    trans_know (defaultdict): A defaultdict object containing the transition parameters for the know_net model.
    """
    # Transition parameters for true_net (only S->E->I->R in Dual net scenario)
    trans_true = defaultdict(list)
    
    # Transition parameters for know_net (only I->T->R in Dual net scenario)
    # If args.dual false, the same net and transition objects are used for both infection and tracing
    trans_know = defaultdict(list) if args.dual else trans_true
    
    # invoke method corresponding to the args.model parameter which will populate accordingly the transition dicts
    globals()[args.model](trans_true, args, exp)
    populate_tracing_trans(trans_know, args, exp)
  
    return trans_true, trans_know


def populate_tracing_trans(trans_know, args, exp):
    """
    Populates the transition rates between compartments in the tracing model.

    Args:
        trans_know (dict): A dictionary of rates between compartments.
        args (argparse.Namespace): An object containing command-line arguments.
        exp (bool): A boolean indicating whether the rates need to be exponentiated or not.

    Returns:
        None
    """
    noncomp = args.noncomp
    # if no noncompliace rate is chosen, separate_traced is inactive or testing is disabled, skip transition T->N
    if args.taur and noncomp and args.separate_traced:
        if args.noncomp_dependtime:
            noncomp_func = get_stateful_sampling_func(
                'expFactorTimesTimeDif', lamda=noncomp, exp=exp)
        else:
            noncomp_func = get_stateless_sampling_func(noncomp, exp)
        add_trans(trans_know, 'T', 'N', noncomp_func)

        
def sir(trans_true, args, exp):
    # add infection rate
    add_trans(trans_true, 'S', 'I', get_stateful_sampling_func('expFactorTimesCount', exp=exp, state='I', lamda=args.beta))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_sampling_func(args.gamma, exp))
    
    
def seir(trans_true, args, exp):
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', get_stateful_sampling_func('expFactorTimesCount', exp=exp, state='I', lamda=args.beta))

    # Next transition is network independent (at rate eps) but we keep the same API for sampling at get_next_event time
    add_trans(trans_true, 'E', 'I', get_stateless_sampling_func(args.eps, exp))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_sampling_func(args.gamma, exp))

        
def covid(trans_true, args, exp):
    # mark args.spontan as always True for Covid
    args.spontan = True
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', get_stateful_sampling_func('expFactorTimesCountMultiState',
                                                               states=['Is'], lamda=args.beta, exp=exp, 
                                                               rel_states=['I', 'Ia'], rel=args.rel_beta))
    
    # Transition to presymp with latency epsilon (we denote Ip state as I !)
    add_trans(trans_true, 'E', 'I', get_stateless_sampling_func(args.eps, exp))
    
    # Transisitons from prodromal state I are based on (probability of being asymp x duration of prodromal phase)
    asymp_dur = args.miup * args.pa
    symp_dur = args.miup * (1 - args.pa)
    add_trans(trans_true, 'I', 'Ia', get_stateless_sampling_func(asymp_dur, exp))
    add_trans(trans_true, 'I', 'Is', get_stateless_sampling_func(symp_dur, exp))
    
    # Asymptomatics can only transition to recovered with duration rate gamma
    add_trans(trans_true, 'Ia', 'R', get_stateless_sampling_func(args.gamma, exp))
    
    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
    symp_hosp = args.gamma * args.ph
    symp_rec = args.gamma * (1 - args.ph)
    add_trans(trans_true, 'Is', 'H', get_stateless_sampling_func(symp_hosp, exp))
    add_trans(trans_true, 'Is', 'R', get_stateless_sampling_func(symp_rec, exp))
    
    # Transitions from hospitalized to R or D are based on measurements in Ile-de-France (Age-group dependent!)
    add_trans(trans_true, 'H', 'R', get_stateless_sampling_func(args.lamdahr, exp))
    add_trans(trans_true, 'H', 'D', get_stateless_sampling_func(args.lamdahd, exp))
    

def add_trans(trans, fr, to, func_or_rate, replace=False):
    """
    Add to `trans` a transition from state `fr` to state `to` with a given function or rate.

    Args:
        trans (dict): A dictionary representing the transitions between states as `fr`:(`to`,`func_or_rate`).
        fr (str): The source state.
        to (str): The target state.
        func_or_rate (callable or float): The function or rate associated with the transition.
        replace (bool, optional): Whether to replace an existing transition to `to` if it exists. Defaults to False.
    """
    if func_or_rate is not None:
        if replace:
            trans[fr] = [(to, func_or_rate) if t[0] == to else t for t in trans[fr]]
        else:
            trans[fr].append((to, func_or_rate))