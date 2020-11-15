import math
import random

from collections import defaultdict

from lib.utils import expFactorTimesCount, expFactorTimesCountMultiState, get_stateless_exp_sampler

def get_transitions_for_model(args):
    # Transition parameters for true_net (only S->E->I->R in Dual net scenario)
    trans_true = defaultdict(list)
    
    # Transition parameters for know_net (only I->T->R in Dual net scenario)
    # If args.dual false, the same net and transition objects are used for both infection and tracing
    trans_know = defaultdict(list) if args.dual else trans_true
    
    # invoke method corresponding to the args.model parameter which will populate accordingly the transition dicts
    globals()[args.model](trans_true, trans_know, args)
    
    return trans_true, trans_know

    
def sir(trans_true, trans_know, args):
    # local vars for efficiency
    beta = args.beta
    presample = args.presample
    # add infection rate
    add_trans(trans_true, 'S', 'I', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=beta))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_exp_sampler(args.gamma, presample))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', get_stateless_exp_sampler(args.gammatau, presample))
    
def seir(trans_true, trans_know, args):
    # local vars for efficiency
    beta = args.beta
    presample = args.presample
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=beta))

    # Next transition is network independent (at rate eps) but we keep the same API for sampling at get_next_event time
    add_trans(trans_true, 'E', 'I', get_stateless_exp_sampler(args.eps, presample))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_exp_sampler(args.gamma, presample))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', get_stateless_exp_sampler(args.gammatau, presample))
    
    
def covid(trans_true, trans_know, args):
    # local vars for efficiency
    beta = args.beta
    rel_beta = args.rel_beta
    presample = args.presample
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid, debug=False:  \
              expFactorTimesCountMultiState(net, nid, states=['Is'], lamda=beta, debug=debug, 
                                            rel_states=['I', 'Ia'], rel=rel_beta))
    
    # Transition to presymp with latency epsilon (we denote I = Ip !!!)
    add_trans(trans_true, 'E', 'I', get_stateless_exp_sampler(args.eps, presample))
    
    # Transisitons from prodromal state I are based on (probability of being asymp x duration of prodromal phase)
    asymp_dur = args.miup * args.pa
    symp_dur = args.miup * (1 - args.pa)
    add_trans(trans_true, 'I', 'Ia', get_stateless_exp_sampler(asymp_dur, presample))
    add_trans(trans_true, 'I', 'Is', get_stateless_exp_sampler(symp_dur, presample))
    
    # Asymptomatics can only transition to recovered with duration rate gamma
    add_trans(trans_true, 'Ia', 'R', get_stateless_exp_sampler(args.gamma, presample))
    
    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
    hosp_rec = args.gamma * args.ph
    hosp_ded = args.gamma * (1 - args.ph)
    add_trans(trans_true, 'Is', 'R', get_stateless_exp_sampler(hosp_rec, presample))
    add_trans(trans_true, 'Is', 'H', get_stateless_exp_sampler(hosp_ded, presample))
    
    # Transitions from hospitalized to R or D are based on measurements in Ile-de-France (Age-group dependent!)
    add_trans(trans_true, 'H', 'R', get_stateless_exp_sampler(args.lamdahr, presample))
    add_trans(trans_true, 'H', 'D', get_stateless_exp_sampler(args.lamdahd, presample))


def add_trans(trans, fr, to, func):
    if func is not None:
        trans[fr].append((to, func))