from collections import defaultdict
import numpy as np
import json

from lib.utils import Event, NumpyEncoder, get_overlap_for_z, get_boxplot_statistics

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
        
    def full_summary(self, splits=1000, printit=False):
        """
        This produces a full summary of the epidemic simulations
        
        splits : number of time intervals
        printit : whether to print the summary to stdout
        """
        
        summary = defaultdict(dict)
        summary['args'] = vars(self.args).copy()
        summary['args']['r0'] = self.args.beta / self.args.gamma
        
        if self.args.dual:
            # If dual=True, true overlap is EITHER the inputted overlap OR (k-zrem)/(k-zadd)
            summary['args']['true-overlap'] = \
                get_overlap_for_z(args.k, args.zadd, args.zrem) if self.args.overlap is None else self.args.overlap
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
                                
            for ser_index in range(len_series):
                # current result in the series_to_sum array
                ser = series_to_sum[ser_index]
                # counter var which will be used to index the current result in the series at a specific time slot
                serI = 1
                for j in range(splits):
                    # increment until the upper time limit of the j-th split - i.e. time_range_limits[j]
                    # is exceeded by an event (and therefore the event will be part of block j + 1)
                    while serI < len(ser) and time_range_limits[j] > ser[serI].time:
                        serI += 1
                    # get last index of the j'th split
                    last_idx = serI - 1
                    # number of infected in the current time frame is the total number of I (includes Ia and Is) and E
                    accumulator[0][j][ser_index] = ser[last_idx].nI + ser[last_idx].nE
                    accumulator[1][j][ser_index] = ser[last_idx].totalInfected
                    accumulator[2][j][ser_index] = ser[last_idx].totalTraced
                    accumulator[3][j][ser_index] = ser[last_idx].totalRecovered
                    accumulator[4][j][ser_index] = ser[last_idx].tracingEffortRandom
                    accumulator[5][j][ser_index] = ser[last_idx].tracingEffortContact[0]
                    accumulator[6][j][ser_index] = ser[last_idx].nH
                    accumulator[7][j][ser_index] = ser[last_idx].totalHospital
                    accumulator[8][j][ser_index] = ser[last_idx].totalDeath
                    accumulator[9][j][ser_index] = ser[last_idx].totalInfectious
                    accumulator[10][j][ser_index] = ser[last_idx].totalFalseTraced
                    accumulator[11][j][ser_index] = ser[last_idx].totalFalsePositive
                    accumulator[12][j][ser_index] = ser[last_idx].totalNonCompliant

                
                # Get Peak of Infection and time of peak
                this_max = ser[0].nI + ser[0].nE
                this_timeofmax = ser[0].time
                for event in ser:
                    if this_max < event.nI + event.nE:
                        this_max = event.nI + event.nE
                        this_timeofmax = event.time    
                i_max.append(this_max)
                i_timeofmax.append(this_timeofmax)                
                
                # Get Peak of Hospitalized and time of peak
                this_max = ser[0].nH
                this_timeofmax = ser[0].time
                for event in ser:
                    if this_max < event.nH:
                        this_max = event.nH
                        this_timeofmax = event.time
                h_max.append(this_max)
                h_timeofmax.append(this_timeofmax)
            
            
            ###############
            # compute averages and other statistics for the over-time simulation results
            stats_for_timed_parameters = []
            for i in range(num_vars):
                stats_for_timed_parameters.append(get_boxplot_statistics(accumulator[i], axis=1))
            stats_for_timed_parameters = np.array(stats_for_timed_parameters)
                
            # compute averages and other statistics for the peak simulation results
            stats_for_max_inf = get_boxplot_statistics(i_max)[0]
            stats_for_timeofmax_inf = get_boxplot_statistics(i_timeofmax)[0]
            stats_for_max_hos = get_boxplot_statistics(h_max)[0]
            stats_for_timeofmax_hos = get_boxplot_statistics(h_timeofmax)[0]
            
            
            ##############
            # update averages dictionary for the current parameter value
            current = summary[param]
            current['time'] = time_range_limits
            
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

        if printit:
            print(json.dumps(summary, cls=NumpyEncoder))

        return summary