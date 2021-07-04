from collections import defaultdict, Counter
import numpy as np
import json

import lib.utils as ut
from lib.utils import get_statistics


# Class to hold stats events
class StatsEvent(ut.Event):
    pass


class StatsProcessor():
    
    def __init__(self, args=None):
        self.args = args
        self.events = []
        self.sim_summary = defaultdict(dict)
        self.param_res = {}
    
    def status_at_time(self, **kwargs):
        self.events.append(StatsEvent(**kwargs))
        
    def status_at_sim_finish(self, inet, itr):
        self.sim_summary[inet][itr] = self.events
        self.events = []
        
    def results_for_param(self, param):
        self.param_res[param] = self.sim_summary
        self.sim_summary = defaultdict(dict)
        
    def __getitem__(self, item):
        return self.param_res[item]
        
    def full_summary(self, splits=1000, printit=0, r_window=7):
        """
        This produces a full summary of the epidemic simulations
        
        splits : number of time intervals
        printit : whether to print the summary to stdout 
            0=no print, 1=print w/out the full predef network (if any), 2=print all, incl full predef network (if any)
        """
        # local for efficiency
        args = self.args
        efforts = args.efforts
        # turn first_inf into an absolute number if it's a percentage by this point (happens only if predefined net with no nid)
        first_inf = args.first_inf = int(args.first_inf if args.first_inf >= 1 else args.first_inf * args.netsize)
        
        summary = defaultdict(dict)
        summary['args'] = vars(args).copy()
        
        if args.model == 'covid':
            # total time of infectiousness (presymp + symp/asymp duration)
            infectious_time_rate = 1 / (1/args.miup + 1/args.gamma)
        else: # in SIR and SEIR the base rates are directly indicative of the transmission/recovery
            infectious_time_rate = args.gamma
            
        contacts_scaler = args.netsize if args.nettype == 'complete' else args.k
        # basic R0 is scaled by the average number of contacts (since the transmission rate is also scaled)
        summary['args']['r0'] = contacts_scaler * args.beta / infectious_time_rate
        
        if args.dual:
            if args.maintain_overlap:
                # If dual=True, true overlap is EITHER the inputted overlap OR (k-zrem)/(k+zadd)
                summary['args']['true-overlap'] = \
                    ut.get_overlap_for_z(args.k, args.zadd, args.zrem) if args.overlap is None else args.overlap
            else:
                summary['args']['true-overlap'] = -1 # this effectively means the overlap was ignored
                
            if args.dual == 2 and args.maintain_overlap_two:
                summary['args']['true-overlap-two'] = \
                    ut.get_overlap_for_z(args.k, args.zadd_two, args.zrem_two) if args.overlap_two is None else args.overlap_two
            else:
                summary['args']['true-overlap-two'] = -1
        else:
            # when single network run (i.e. dual=False), the true overlap is 1 since all the dual configs are ignored
            summary['args']['true-overlap'] = 1
            # we also mark the true overlap of the second dual network as effectively inexistent/ignored
            summary['args']['true-overlap-two'] = -1
        
        
        for param, results_for_param in self.param_res.items():
            
            # transform from defaultdict of dimensions inet, itr, num_events to list of dimension inet * itr , num_events
            series_to_sum = [sim_events for inet in results_for_param for sim_events in results_for_param[inet].values()]
            # the number of events for avg and var calculation
            len_series = len(series_to_sum)
        
            max_time = float('-inf')
            for ser in series_to_sum:
                max_time = max(max_time, ser[-1].time)
                            
            # will hold all max and timeofmax for infected and hospitalized for calculating avg and std at the end
            i_max = []
            i_timeofmax = []
            h_max = []
            h_timeofmax = []
            
            # time range right-limits
            time_range_limits = np.zeros(splits)
            # set upper time limits for each accumulator[:][j] pile
            # accumulator[:][0] is reserved for the initial configuration of the simulation, so only (splits - 1) slots remain available
            available_time_splits = splits - 1
            for j in range(1, splits):
                time_range_limits[j] = max_time * j / available_time_splits

            # number of different vars computed across time
            num_vars = 13
            # holds simulation result parameters over time
            accumulator = np.zeros((num_vars, splits, len_series))
            # set the first infected number for time 0 (second-dim-id = 0); indices in first dim: active-inf, total-inf, total-infectious
            accumulator[np.ix_([0, 1, 9], [0], range(len_series))] = first_inf
            # indexes of early stopped
            idx_early_stopped = []
            # array of arrays containing multiple Reff/growth_rates measurements sampled every r_window
            growth_series = []
            active_growth_series = []
                                
            for ser_index, ser in enumerate(series_to_sum):
                # if the overall infected remained smaller than the no. of initial infected + selected margin, account as earlystop
                if ser[-1].totalInfected <= args.first_inf + args.earlystop_margin:
                    idx_early_stopped.append(ser_index)
                # counter var which will be used to index the current result in the series at a specific time slot
                serI = 1
                # iterate over "time splits"
                for j in range(1, splits):
                    # increment until the upper time limit of the j-th split - i.e. time_range_limits[j]
                    # is exceeded by an event (and therefore the event will be part of block j + 1)
                    while serI < len(ser) and time_range_limits[j] > ser[serI].time:
                        serI += 1
                    # get last index of the j'th split
                    last_idx = ser[serI - 1]
                    # number of infected in the current time frame is the total number of I (includes Ia and Is) and E
                    accumulator[0][j][ser_index] = last_idx.nI + last_idx.nE
                    accumulator[1][j][ser_index] = last_idx.totalInfected
                    accumulator[2][j][ser_index] = last_idx.totalTraced
                    accumulator[3][j][ser_index] = last_idx.totalRecovered
                    if efforts:
                        accumulator[4][j][ser_index] = last_idx.tracingEffortRandom
                        accumulator[5][j][ser_index] = last_idx.tracingEffortContact[0]
                    accumulator[6][j][ser_index] = last_idx.nH
                    accumulator[7][j][ser_index] = last_idx.totalHospital
                    accumulator[8][j][ser_index] = last_idx.totalDeath
                    accumulator[9][j][ser_index] = last_idx.totalInfectious
                    accumulator[10][j][ser_index] = last_idx.totalFalseTraced
                    accumulator[11][j][ser_index] = last_idx.totalFalseNegative
                    accumulator[12][j][ser_index] = last_idx.totalNonCompliant

                
                # Get Peak of Infection, Peak of Hospitalization, Time of peaks, and growth rate for r_window
                # 'reference_' vars are for growths, '_max' vars are for peaks
                this_i_max = reference_past_active_inf = reference_past_total_inf = first_inf
                this_h_max = 0
                this_i_timeofmax = this_h_timeofmax = reference_past_time = 0
                growth = []
                active_growth = []
                
                for event in ser:
                    event_time = event.time
                    active_infected = event.nI + event.nE
                    active_hosp = event.nH
                    total_infected = event.totalInfected
                    # Block which will eventually calculate peak of infection
                    if this_i_max < active_infected:
                        this_i_max = active_infected
                        this_i_timeofmax = event_time
                    # Block which will eventually calculate peak of hospitalization
                    if this_h_max < active_hosp:
                        this_h_max = active_hosp
                        this_h_timeofmax = event_time
                    # Calculate Reff for a specific time period if there were any active infected at the previous time point
                    if reference_past_active_inf and event_time >= r_window + reference_past_time:
                        new_infections = total_infected - reference_past_total_inf
                        # growth_rate = new_cases / active_cases_past
                        growth.append(new_infections / reference_past_active_inf)
                        # active_growth_rate = active_cases_now / active_cases_past
                        active_growth.append(active_infected / reference_past_active_inf)
                        # change reference past measurement to use for the next Reff computation
                        reference_past_total_inf = total_infected
                        reference_past_time = event_time
                        reference_past_active_inf = active_infected
                        
                i_max.append(this_i_max)
                i_timeofmax.append(this_i_timeofmax)
                h_max.append(this_h_max)
                h_timeofmax.append(this_h_timeofmax)
                # note this will be a list of lists
                growth_series.append(growth)
                active_growth_series.append(active_growth)
                
            # indexes to remove from the alternative mean_wo and std_wo calculation (only if option selected from args)
            without_idx = idx_early_stopped if args.avg_without_earlystop else None
                        
            ###############
            # compute averages and other statistics for all simulation result parameters over time
            stats_for_timed_parameters = []
            stats_for_laststamp = []
            for i in range(num_vars):
                current_stats = accumulator[i]
                stats_for_timed_parameters.append(get_statistics(current_stats, compute='mean', axis=1, avg_without_idx=without_idx))
                stats_for_laststamp.append(get_statistics(current_stats[-1], compute='all', avg_without_idx=without_idx)[0])
                                
            # compute averages and other statistics for the peak simulation results
            stats_for_max_inf = get_statistics(i_max, compute='all', avg_without_idx=without_idx)[0]
            stats_for_timeofmax_inf = get_statistics(i_timeofmax, compute='mean+wo', avg_without_idx=without_idx)[0]
            stats_for_max_hos = get_statistics(h_max, compute='all', avg_without_idx=without_idx)[0]
            stats_for_timeofmax_hos = get_statistics(h_timeofmax, compute='mean+wo', avg_without_idx=without_idx)[0]
            
            # compute averages and other stats for growth rates
            stats_for_growth = get_statistics(ut.pad_2d_list_variable_len(growth_series),
                                              compute='mean+wo', avg_without_idx=without_idx)
            stats_for_active_growth = get_statistics(ut.pad_2d_list_variable_len(active_growth_series),
                                                     compute='mean+wo', avg_without_idx=without_idx)
            
            ##############
            # update averages dictionary for the current parameter value
            # the key for results in the summary dictionary will be 'res' if the simulations ran for one parameter value only
            # otherwise the key will be the actual parameter value
            key_for_res = 'res' if len(self.param_res) == 1 else param
            current = summary[key_for_res]
            current['time'] = time_range_limits
            current['early-stopped'] = len(idx_early_stopped)
            
            current['average-infected'] = stats_for_timed_parameters[0]
            current['average-max-infected'] = stats_for_max_inf
            current['average-time-of-max-infected'] = stats_for_timeofmax_inf
            
            current['average-total-infected'] = stats_for_timed_parameters[1]
            current['average-overall-infected'] = stats_for_laststamp[1]
            
            current['average-total-traced'] = stats_for_timed_parameters[2]
            current['average-overall-traced'] = stats_for_laststamp[2]
            
            current['average-total-recovered'] = stats_for_timed_parameters[3]
            current['average-overall-recovered'] = stats_for_laststamp[3]
            
            if efforts:
                current['average-effort-random'] = stats_for_timed_parameters[4]
                current['average-effort-contact'] = stats_for_timed_parameters[5]
            
            current['average-hospital'] = stats_for_timed_parameters[6]
            current['average-max-hospital'] = stats_for_max_hos
            current['average-time-of-max-hospital'] = stats_for_timeofmax_hos

            current['average-total-hospital'] = stats_for_timed_parameters[7]
            current['average-overall-hospital'] = stats_for_laststamp[7]
            
            current['average-total-death'] = stats_for_timed_parameters[8]
            current['average-overall-death'] = stats_for_laststamp[8]

            current['average-total-infectious'] = stats_for_timed_parameters[9]
            current['average-overall-infectious'] = stats_for_laststamp[9]

            current['average-total-false-traced'] = stats_for_timed_parameters[10]
            current['average-overall-false-traced'] = stats_for_laststamp[10]
            # the total of correctly traced individuals is the difference between the overall traced and the falsely traced nodes
            current['average-overall-true-traced'] = dict(Counter(current['average-overall-traced']) - Counter(current['average-overall-false-traced']))
            
            current['average-total-false-negative'] = stats_for_timed_parameters[11]
            current['average-overall-false-negative'] = stats_for_laststamp[11]
            
            current['average-total-noncompliant'] = stats_for_timed_parameters[12]
            current['average-overall-noncompliant'] = stats_for_laststamp[12]
            
            # Good moment for quickly debugging how well tracing fared overall
#             print('Overall traced: ', current['average-overall-traced'])
            
            # This is based on Tsimring and Huerta 2002
            current['r-trace'] = contacts_scaler * args.beta / (infectious_time_rate + args.beta + args.taur *
                                                                (1 + current['average-overall-true-traced']['mean'] / current['average-overall-infected']['mean']))
            # growth rates based on r_window
            current['growth'] = stats_for_growth
            # for COVID models, we can also compute R_eff during the first period of the simulation from the growth recorded within the same period
            # since we know both the mean and shape of the generation time ditribution
            if args.model == 'covid':
                # we want R to reflect the r_window chosen and be corresponding to the first estimate of the growth
                current['r-eff'] = ut.r_from_growth(stats_for_growth[0]['mean'], method='exp', t=r_window, mean=6.6, shape=1.87, inv_scale=0.28)
            current['active-growth'] = stats_for_active_growth

        if printit:
            # if printit is 1 and nettype is a predefined supplied network (i.e. not a str), 
            # the repr of the net in the summary will be overwritten to reduce the output space
            if printit == 1 and not isinstance(summary['args']['nettype'], str):
                summary['args']['nettype'] = 'predefined'
            print(json.dumps(summary, cls=ut.NumpyEncoder))

        return summary