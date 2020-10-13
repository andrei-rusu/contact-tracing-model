from collections import defaultdict
import numpy as np
import json

from lib.utils import Event, NumpyEncoder

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
        
    def full_summary(self, printit=False):
        """
        This produces a full summary of the simulations run via run.run_mock
        """
        
        summary = defaultdict(dict)
        summary['args'] = vars(self.args).copy()
        summary['args']['r0'] = self.args.beta / self.args.gamma
        summary['args']['true-overlap'] = self.args.overlap if self.args.dual else 1
        
        for param, results_for_param in self.param_res.items():
            
            # transform from defaultdict of dimensions inet, itr, num_events to list of dimension inet * itr , num_events
            series_to_sum = [sim_events for inet in results_for_param for sim_events in results_for_param[inet].values()]
        
            max_time = float('-inf')
            for ser in series_to_sum:
                max_time = max(max_time, ser[-1].time)
                
            split = 1000
            
            avg_max = 0
            avg_timeofmax = 0
            avg_overallinf = 0
            avg_h_max = 0
            avg_h_timeofmax = 0
            avg_h_overallhosp = 0
            avg = np.zeros((10, split))
                        
            for i in range(split):
                avg[0][i] = max_time * (i + 1) / split
                                
            for ser in series_to_sum:
                serI = 1
                for j in range(split):
                    # increment until the upper time limit of the j-th split - i.e. avg[0][j]
                    # is exceeded by an event (and therefore the event will be part of block j + 1)
                    while serI < len(ser) and avg[0][j] > ser[serI].time:
                        serI += 1
                    # get last index of the j'th split
                    last_idx = serI - 1
                    avg[1][j] += ser[last_idx].nI + ser[last_idx].nE
                    avg[2][j] += ser[last_idx].totalInfected
                    avg[3][j] += ser[last_idx].totalTraced
                    avg[4][j] += ser[last_idx].totalRecovered
                    avg[5][j] += ser[last_idx].tracingEffortRandom
                    avg[6][j] += ser[last_idx].tracingEffortContact
                    avg[7][j] += ser[last_idx].nH
                    avg[8][j] += ser[last_idx].totalHospital
                    avg[9][j] += ser[last_idx].totalDeath

                
                # Get Peak of Infection and total overall infected
                this_max = ser[0].nI + ser[0].nE
                this_timeofmax = ser[0].time
                for event in ser:
                    if this_max < event.nI + event.nE:
                        this_max = event.nI + event.nE
                        this_timeofmax = event.time    
                avg_max += this_max
                avg_timeofmax += this_timeofmax
                avg_overallinf += ser[-1].totalInfected
                
                
                # Get Peak of Hospitalized and total overall hospitalized
                this_max = ser[0].nH
                this_timeofmax = ser[0].time
                for event in ser:
                    if this_max < event.nH:
                        this_max = event.nH
                        this_timeofmax = event.time
                        
                avg_h_max += this_max
                avg_h_timeofmax += this_timeofmax
                avg_h_overallhosp += ser[-1].totalHospital


            # normalize by the number of events
            len_series = len(series_to_sum)
            avg[1:] /= len_series
            avg_max /= len_series
            avg_timeofmax /= len_series
            avg_overallinf /= len_series
            avg_h_max /= len_series
            avg_h_timeofmax /= len_series
            avg_h_overallhosp /= len_series
            
            # update averages dictionary
            current = summary[param]
            current['time'] = avg[0]
            current['average-infected'] = avg[1]
            current['average-max-infected'] = avg_max
            current['average-time-of-max-infected'] = avg_timeofmax
            current['average-overall-infected'] = avg_overallinf
            current['average-total-infected'] = avg[2]
            current['average-total-traced'] = avg[3]
            current['average-total-recovered'] = avg[4]
            current['average-effort-random'] = avg[5]
            current['average-effort-contact'] = avg[6]
            current['average-effort-total'] = avg[5] + avg[6]
            current['average-hospital'] = avg[7]
            current['average-max-hospital'] = avg_h_max
            current['average-time-of-max-hospital'] = avg_h_timeofmax
            current['average-total-hospital'] = avg_h_overallhosp
            current['average-total-death'] = avg[9]

        if printit:
            print(json.dumps(summary, cls=NumpyEncoder))

        return summary
           
            