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
    add_trans(trans_true, 'S', 'I', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_exp_sampler(args.gamma, args.presample))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', get_stateless_exp_sampler(args.gammatau, args.presample))
    
def seir(trans_true, trans_know, args):
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta))

    # Next transition is network independent (at rate eps) but we keep the same API for sampling at get_next_event time
    add_trans(trans_true, 'E', 'I', get_stateless_exp_sampler(args.eps, args.presample))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', get_stateless_exp_sampler(args.gamma, args.presample))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', get_stateless_exp_sampler(args.gammatau, args.presample))
    
    
def covid(trans_true, trans_know, args):
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid, debug=False:  \
              expFactorTimesCountMultiState(net, nid, states=['Is'], lamda=args.beta, debug=debug, 
                                            rel_states=['I', 'Ia'], rel=args.rel_beta))
    
    # Transition to presymp with latency epsilon (we denote I = Ip !!!)
    add_trans(trans_true, 'E', 'I', get_stateless_exp_sampler(args.eps, args.presample))
    
    # Transisitons from prodromal state I are based on (probability of being asymp x duration of prodromal phase)
    asymp_dur = args.miup * args.pa
    symp_dur = args.miup * (1 - args.pa)
    add_trans(trans_true, 'I', 'Ia', get_stateless_exp_sampler(asymp_dur, args.presample))
    add_trans(trans_true, 'I', 'Is', get_stateless_exp_sampler(symp_dur, args.presample))
    
    # Asymptomatics can only transition to recovered with duration rate gamma
    add_trans(trans_true, 'Ia', 'R', get_stateless_exp_sampler(args.gamma, args.presample))
    
    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
    hosp_rec = args.gamma * args.ph
    hosp_ded = args.gamma * (1 - args.ph)
    add_trans(trans_true, 'Is', 'R', get_stateless_exp_sampler(hosp_rec, args.presample))
    add_trans(trans_true, 'Is', 'H', get_stateless_exp_sampler(hosp_ded, args.presample))
    
    # Transitions from hospitalized to R or D are based on measurements in Ile-de-France (Age-group dependent!)
    add_trans(trans_true, 'H', 'R', get_stateless_exp_sampler(args.lamdahr, args.presample))
    add_trans(trans_true, 'H', 'D', get_stateless_exp_sampler(args.lamdahd, args.presample))


def add_trans(trans, fr, to, func):
    if func is not None:
        trans[fr].append((to, func))