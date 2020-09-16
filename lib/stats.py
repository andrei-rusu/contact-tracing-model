from collections import defaultdict
import numpy as np

from lib.utils import Event

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
        
    def full_summary(self):
        """
        This produces a full summary of the simulations run via run.run_mock
        """
        
        summary = defaultdict(dict)
        summary['args'] = vars(self.args).copy()
        summary['args']['r0'] = self.args.beta / self.args.gamma
        summary['args']['true-overlap'] = self.args.overlap if self.args.dual else 1
        
        for param, res in self.param_res.items():
            
            # transform from defaultdict of dimensions inet, itr, num_events to list of dimension inet * itr , num_events
            series_to_sum = [sim_events for inet in res for itr, sim_events in res[inet].items()]
        
            max_time = float('-inf')
            for ser in series_to_sum:
                max_time = max(max_time, ser[-1].time)
                
            split = 1000
            
            avg_max = 0
            avg_timeofmax = 0
            avg_overallinf = 0
            avg = np.zeros((7, split))
                        
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
                    avg[1][j] += ser[last_idx].nI
                    avg[2][j] += ser[last_idx].totalInfected
                    avg[3][j] += ser[last_idx].totalTraced
                    avg[4][j] += ser[last_idx].totalRemoved
                    avg[5][j] += ser[last_idx].tracingEffortRandom
                    avg[6][j] += ser[last_idx].tracingEffortContact
                    
                this_max = ser[0].nI
                this_timeofmax = ser[0].time
                for event in ser:
                    if this_max < event.nI:
                        this_max = event.nI
                        this_timeofmax = event.time
                        
                avg_max += this_max
                avg_timeofmax += this_timeofmax
                avg_overallinf += ser[-1].totalInfected


            # normalize by the number of events
            len_series = len(series_to_sum)
            avg[1:] /= len_series
            avg_max /= len_series
            avg_timeofmax /= len_series
            avg_overallinf /= len_series
            
            # update averages dictionary
            summary[param]['time'] = avg[0]
            summary[param]['average-infected'] = avg[1]
            summary[param]['average-max-infected'] = avg_max
            summary[param]['average-time-of-max-infected'] = avg_timeofmax
            summary[param]['average-overall-infected'] = avg_overallinf
            summary[param]['average-total-infected'] = avg[2]
            summary[param]['average-total-traced'] = avg[3]
            summary[param]['average-total-removed'] = avg[4]
            summary[param]['average-effort-random'] = avg[5]
            summary[param]['average-effort-contact'] = avg[6]
            summary[param]['average-effort-total'] = avg[5] + avg[6]

        return summary
           
            