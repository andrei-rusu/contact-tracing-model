from collections import defaultdict
import numpy as np
import json

from lib.utils import Event, NumpyEncoder, get_overlap_for_z, get_boxplot_statistics, pad_2d_list_variable_len

# Class to hold stats events
class StatsEvent(Event):
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
        """
        # local for efficiency
        args = self.args
        
        summary = defaultdict(dict)
        summary['args'] = vars(args).copy()
        
        if args.model == 'covid':
            # total time of infectiousness (presymp + symp/asymp duration)
            infectious_time_rate = 1 / (1/args.miup + 1/args.gamma) 
            # infection rate of first infected (scaled by rel infectiousness since first infected are Ip)
            initial_inf_rate = args.beta * args.rel_beta 
        else: # in SIR and SEIR the base rates are directly indicative of the transmission/recovery
            infectious_time_rate = args.gamma
            initial_inf_rate = args.beta
            
        # basic R0 is scaled by the average number of contacts (since the transmission rate is also scaled)
        contacts_scaler = args.netsize if args.nettype == 'complete' else args.k
        summary['args']['r0'] = contacts_scaler * args.beta / infectious_time_rate
        # also keep track of the initial R0 for the simulation i.e. for the first infected (all are Ip at start)
        summary['args']['r0-first'] = args.first_inf * initial_inf_rate / infectious_time_rate
        
        if args.dual:
            # If dual=True, true overlap is EITHER the inputted overlap OR (k-zrem)/(k-zadd)
            summary['args']['true-overlap'] = \
                get_overlap_for_z(args.k, args.zadd, args.zrem) if args.overlap is None else args.overlap
        else:
            # when single network run (i.e. dual=False), the true overlap is 1 since all the dual configs are ignored
            summary['args']['true-overlap'] = 1
        
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
            
            # time range right limits
            time_range_limits = np.zeros(splits)
            # set upper time limits for each accumulator[:][j] pile
            for j in range(splits):
                time_range_limits[j] = max_time * (j + 1) / splits

            # number of different vars computed across time
            num_vars = 13
            # holds simulation result parameters over time
            accumulator = np.zeros((num_vars, splits, len_series))
            # indexes of early stopped
            idx_early_stopped = []
            # array of arrays containing multiple Reff/growth_rates measurements sampled every r_window
            r_eff_series = []
            growth_series = []
                                
            for ser_index, ser in enumerate(series_to_sum):
                # if the overall infected remained smaller than the no. of initial infected + selected margin, account as earlystop
                if ser[-1].totalInfected <= args.first_inf + args.earlystop_margin:
                    idx_early_stopped.append(ser_index)
                # counter var which will be used to index the current result in the series at a specific time slot
                serI = 1
                for j in range(splits):
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
                    accumulator[4][j][ser_index] = last_idx.tracingEffortRandom
                    accumulator[5][j][ser_index] = last_idx.tracingEffortContact[0]
                    accumulator[6][j][ser_index] = last_idx.nH
                    accumulator[7][j][ser_index] = last_idx.totalHospital
                    accumulator[8][j][ser_index] = last_idx.totalDeath
                    accumulator[9][j][ser_index] = last_idx.totalInfectious
                    accumulator[10][j][ser_index] = last_idx.totalFalseTraced
                    accumulator[11][j][ser_index] = last_idx.totalFalsePositive
                    accumulator[12][j][ser_index] = last_idx.totalNonCompliant

                
                # get first event -> needed to compute peaks and Reff
                first_event = ser[0]
                r_eff = []
                growth = []

                # Get Peak of Infection, Peak of Hospitalization, Time of peaks, and Reff for r_window
                # reference_ vars are for Reff, _max vars are for peaks
                this_i_max = reference_past_active_inf = first_event.nI + first_event.nE
                this_h_max = first_event.nH
                reference_past_total_inf = first_event.totalInfected
                this_i_timeofmax = this_h_timeofmax = reference_past_time = first_event.time
                for event in ser:
                    event_time = event.time
                    active_infected = event.nI + event.nE
                    active_infectious = event.nI
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
                        # we consider Reff = new_cases / active_cases_past
                        r_eff.append(new_infections / reference_past_active_inf)
                        # the growth_rate = active_cases_now / active_cases_past
                        growth.append(active_infected /  reference_past_active_inf)
                        # change reference past measurement to use for the next Reff computation
                        reference_past_total_inf = total_infected
                        reference_past_time = event_time
                        reference_past_active_inf = active_infected                            
                        
                i_max.append(this_i_max)
                i_timeofmax.append(this_i_timeofmax)                
                h_max.append(this_h_max)
                h_timeofmax.append(this_h_timeofmax)
                # note this will be a list of lists
                r_eff_series.append(r_eff)
                growth_series.append(growth)
                
            # indexes to remove from the alternative mean_wo and std_wo calculation (only if option selected from args)
            without_idx = idx_early_stopped if args.avg_without_earlystop else None
                        
            ###############
            # compute averages and other statistics for the over-time simulation results
            stats_for_timed_parameters = []
            for i in range(num_vars):
                stats_for_timed_parameters.append(get_boxplot_statistics(accumulator[i], axis=1, avg_without_idx=without_idx))
            stats_for_timed_parameters = np.array(stats_for_timed_parameters)
                
            # compute averages and other statistics for the peak simulation results
            stats_for_max_inf = get_boxplot_statistics(i_max, avg_without_idx=without_idx)[0]
            stats_for_timeofmax_inf = get_boxplot_statistics(i_timeofmax, avg_without_idx=without_idx)[0]
            stats_for_max_hos = get_boxplot_statistics(h_max, avg_without_idx=without_idx)[0]
            stats_for_timeofmax_hos = get_boxplot_statistics(h_timeofmax, avg_without_idx=without_idx)[0]
            
            # compute averages and other stats for r_eff's
            stats_for_r_eff = get_boxplot_statistics(pad_2d_list_variable_len(r_eff_series), axis=0, avg_without_idx=without_idx)
            stats_for_growth = get_boxplot_statistics(pad_2d_list_variable_len(growth_series), axis=0, avg_without_idx=without_idx)
            
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
            current['average-overall-infected'] = stats_for_timed_parameters[1][-1]
            
            current['average-total-traced'] = stats_for_timed_parameters[2]
            current['average-overall-traced'] = stats_for_timed_parameters[2][-1]
            
            current['average-total-recovered'] = stats_for_timed_parameters[3]
            current['average-overall-recovered'] = stats_for_timed_parameters[3][-1]
            
            current['average-effort-random'] = stats_for_timed_parameters[4]
            current['average-effort-contact'] = stats_for_timed_parameters[5]
            
            current['average-hospital'] = stats_for_timed_parameters[6]
            current['average-max-hospital'] = stats_for_max_hos
            current['average-time-of-max-hospital'] = stats_for_max_hos

            current['average-total-hospital'] = stats_for_timed_parameters[7]
            current['average-overall-hospital'] = stats_for_timed_parameters[7][-1]
            
            current['average-total-death'] = stats_for_timed_parameters[8]
            current['average-overall-death'] = stats_for_timed_parameters[8][-1]

            current['average-total-infectious'] = stats_for_timed_parameters[9]
            current['average-overall-infectious'] = stats_for_timed_parameters[9][-1]

            current['average-total-false-traced'] = stats_for_timed_parameters[10]
            current['average-overall-false-traced'] = stats_for_timed_parameters[10][-1]
            
            current['average-total-false-positive'] = stats_for_timed_parameters[11]
            current['average-overall-false-positive'] = stats_for_timed_parameters[11][-1]
            
            current['average-total-noncompliant'] = stats_for_timed_parameters[12]
            current['average-overall-noncompliant'] = stats_for_timed_parameters[12][-1]
            
            # This is based on Tsimring and Huerta 2002
            current['r-trace'] = contacts_scaler * args.beta / (infectious_time_rate + args.beta + args.taur * \
                                        (1 + current['average-overall-traced']['mean']/current['average-overall-infected']['mean']))
            # Reff based on r_window
            current['r-eff'] = stats_for_r_eff
            current['growth'] = stats_for_growth

        if printit:
            print(json.dumps(summary, cls=NumpyEncoder))

        return summary