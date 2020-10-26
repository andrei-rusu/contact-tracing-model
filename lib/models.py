import math
import random

from collections import defaultdict

from lib.utils import expFactorTimesCount, expFactorTimesCountMultiState

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
    add_trans(trans_true, 'S', 'I', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta, base=0))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', lambda net, nid: -(math.log(random.random()) / args.gamma))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', lambda net, nid: -(math.log(random.random()) / args.gammatau))
    
def seir(trans_true, trans_know, args):
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid: expFactorTimesCount(net, nid, state='I', lamda=args.beta, base=0))

    # Next transition is network independent (at rate eps) but we keep the same API for sampling at get_next_event time
    add_trans(trans_true, 'E', 'I', lambda net, nid: -(math.log(random.random()) / args.eps))
    
    if args.spontan:
        # allow spontaneuous recovery (without tracing) with rate gamma
        add_trans(trans_true, 'I', 'R', lambda net, nid: -(math.log(random.random()) / args.gamma))
        
    # Recovery for traced nodes is network independent at rate gammatau
    add_trans(trans_know, 'T', 'R', lambda net, nid: -(math.log(random.random()) / args.gammatau))
    
    
def covid(trans_true, trans_know, args):
    # Infections spread based on true_net connections depending on nid
    add_trans(trans_true, 'S', 'E', lambda net, nid:  \
              expFactorTimesCountMultiState(net, nid, states=['Is'], lamda=args.beta, base=0, rel_states=['I', 'Ia'], rel=args.rel_beta))
    
    # Transition to presymp with latency epsilon (we denote I = Ip !!!)
    add_trans(trans_true, 'E', 'I', lambda net, nid: -(math.log(random.random()) / args.eps))
    
    # Transisitons from prodromal state I are based on (probability of being asymp x duration of prodromal phase)
    asymp_dur = args.miup * args.pa
    symp_dur = args.miup * (1 - args.pa)
    add_trans(trans_true, 'I', 'Ia', lambda net, nid: -(math.log(random.random()) / asymp_dur))
    add_trans(trans_true, 'I', 'Is', lambda net, nid: -(math.log(random.random()) / symp_dur))
    
    # Asymptomatics can only transition to recovered with duration rate gamma
    add_trans(trans_true, 'Ia', 'R', lambda net, nid: -(math.log(random.random()) / args.gamma))
    
    # Symptomatics can transition to either recovered or hospitalized based on duration gamma and probability ph (Age-group dependent!)
    hosp_rec = args.gamma * args.ph
    hosp_ded = args.gamma * (1 - args.ph)
    add_trans(trans_true, 'Is', 'R', lambda net, nid: -(math.log(random.random()) / hosp_rec))
    add_trans(trans_true, 'Is', 'H', lambda net, nid: -(math.log(random.random()) / hosp_ded))
    
    # Transitions from hospitalized to R or D are based on measurements in Ile-de-France (Age-group dependent!)
    add_trans(trans_true, 'H', 'R', lambda net, nid: -(math.log(random.random()) / args.lamdahr))
    add_trans(trans_true, 'H', 'D', lambda net, nid: -(math.log(random.random()) / args.lamdahd))

def add_trans(trans, fr, to, func):
    if func is not None:
        trans[fr].append((to, func))