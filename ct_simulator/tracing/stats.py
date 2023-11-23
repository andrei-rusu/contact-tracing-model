import os
import numpy as np
import json
from collections import defaultdict, Counter

from . import utils as ut
from .utils import get_statistics


STATS_MEAN_OPT = 'mean'
STATS_MEAN_WO_OPT = 'mean+wo'
STATS_ALL_OPT = 'all'


class StatsEvent(ut.Event):
    """
    A custom event class for tracking statistics recording events.

    This class inherits from the `ut.Event` class and can be used to track various statistics
    throughout the program. It can be subclassed to add additional functionality as needed.
    """
    pass


class StatsProcessor():
    """
    A class for processing and summarizing statistics for epidemic simulations.

    Attributes:
        args: A dictionary of arguments for the simulation.
        events: A list of StatsEvent objects representing the status of the simulation at various times.
        sim_summary: A defaultdict of dictionaries containing the events for each simulation run.
        param_res: A dictionary of simulation results for each parameter.
    """
    def __init__(self, args=None):
        """
        Initializes a new StatsProcessor object.

        Args:
            args: A dictionary of arguments for the simulation.
        """
        self.args = args
        self.events = []
        self.sim_summary = defaultdict(dict)
        self.param_res = {}

    def status_at_time(self, **kwargs):
        """
        Adds a new StatsEvent object to the events list.

        Args:
            kwargs: A dictionary of keyword arguments representing the status of the simulation at a given time.
        """
        self.events.append(StatsEvent(**kwargs))

    def status_at_sim_finish(self, inet, itr):
        """
        Adds the events list to the sim_summary dictionary for the given simulation run.

        Args:
            inet: The index of the simulation run.
            itr: The index of the iteration within the simulation run.
        """
        self.sim_summary[inet][itr] = self.events
        self.events = []

    def results_for_param(self, param):
        """
        Adds the sim_summary dictionary to the param_res dictionary for the given parameter.

        Args:
            param: The name of the parameter.
        """
        self.param_res[param] = self.sim_summary
        self.sim_summary = defaultdict(dict)

    def __getitem__(self, item):
        """
        Returns the simulation results for the given parameter.

        Args:
            item: The name of the parameter.

        Returns:
            defaultdict: The simulation results for the given parameter.
        """
        return self.param_res[item]
        
    def full_summary(self, splits=1000, printit=0., r_window=7):
        """
        Produces a full summary of the epidemic simulations.

        Args:
            splits: The number of time intervals (this will effectively split the total time into equal intervals).
            printit: Whether to print the summary to stdout (0=no print, <2=print to stdout, >=2=print to file).
                Special setting: if the decimal part of `printit` is 5, the `args` will not be printed.
            r_window: The size of the window for calculating R_e and growth rates.

        Returns:
            summary (defaultdict): A dictionary containing the simulation results and arguments.
        """
        # local for efficiency
        args = self.args
        efforts = args.efforts
        
        # copy of the args dict that will be used for logging summary
        kwargs = vars(args).copy()
        # remove or convert to str certain large or unnecessary objects from the summary holder
        del kwargs['shared']
        kwargs.pop('is_learning_agent', None)
        kwargs.pop('is_record_agent', None)
        # variable which indicates whether a control agent was used or not
        use_agent = False
        if kwargs['agent'] is not None:
            use_agent = True
            for kwarg in ('ranking_model', 'target_model', 'tb_layout'):
                kwargs['agent'].pop(kwarg, None)
            for kwarg in ('optimizer', 'schedulers'):
                if kwarg in kwargs:
                    kwargs['agent'][kwarg] = str(kwargs['agent'][kwarg])
        # turn first_inf into an absolute number if it's a percentage by this point (happens only if predefined net with no nids set)
        kwargs['first_inf'] = first_inf = int(args.first_inf if args.first_inf >= 1 else args.first_inf * args.netsize)
            
        summary = defaultdict(dict)
        summary['args'] = kwargs
        
        for param, results_for_param in self.param_res.items():
            # transform from defaultdict of dimensions inet, itr, num_events to list of dimension inet * itr , num_events
            # num_events is variable for each (inet, itr) combination
            series_to_sum = [sim_events for inet in results_for_param for sim_events in results_for_param[inet].values()]
            # total number of simulations
            len_series = len(series_to_sum)
        
            max_time = 0
            for ser in series_to_sum:
                if ser:
                    max_time = max(max_time, ser[-1].time)
                            
            # will hold all max and timeofmax for infected and hospitalized for calculating avg and std at the end
            i_max = []
            i_timeofmax = []
            h_max = []
            h_timeofmax = []
            # agent-only metrics
            detected = []
            test_src = []
            test_src_unique = []
            trace_src = []
            trace_src_unique = []
            # array of arrays containing multiple Reff/growth_rates measurements sampled every r_window
            growth_series = []
            active_growth_series = []
            # the inf-trace timelag histograms
            timelag_hist = []
            
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
            # set the first infected number as default for all time splits in all series; indices in first dim: active-inf, total-inf, total-infectious
            accumulator[np.ix_([0, 1, 9], range(splits), range(len_series))] = first_inf
            # indexes of early stopped
            idx_early_stopped = []
            
            for ser_index, ser in enumerate(series_to_sum):
                if ser:
                    last_step = ser[-1]
                    if len(ser) > 1:
                        # if the overall infected remained smaller than the no. of initial infected + selected margin, account as earlystop
                        if last_step.totalInfected <= first_inf + args.earlystop_margin:
                            idx_early_stopped.append(ser_index)
                        # counter var which will be used to index the current result in the series at a specific time slot
                        ser_i = 1
                        # iterate over "time splits"
                        for j in range(1, splits):
                            # increment until the upper time limit of the j-th split - i.e. time_range_limits[j]
                            # is exceeded by an event (and therefore the event will be part of block j)
                            while ser_i < len(ser) - 1 and ser[ser_i].time < time_range_limits[j]:
                                ser_i += 1
                            # get last index of the j'th split
                            last_idx = ser[ser_i]
                            # number of infected in the current time frame is the total number of I (includes Ia and Is) and E
                            accumulator[0][j][ser_index] = last_idx.nI + last_idx.nE
                            accumulator[1][j][ser_index] = last_idx.totalInfected
                            accumulator[2][j][ser_index] = last_idx.totalTraced
                            accumulator[3][j][ser_index] = last_idx.totalRecovered
                            if efforts:
                                accumulator[4][j][ser_index] = last_idx.tracingEffortRandom
                                accumulator[5][j][ser_index] = last_idx.tracingEffortContact[0]
                            accumulator[6][j][ser_index] = last_idx.nH
                            accumulator[7][j][ser_index] = \
                                last_idx.totalHospital if args.model == 'covid' else last_idx.totalRecovered * args.ph
                            accumulator[8][j][ser_index] = \
                                last_idx.totalDeath if args.model == 'covid' else accumulator[7][j][ser_index] * args.lamdahd
                            accumulator[9][j][ser_index] = last_idx.totalInfectious
                            accumulator[10][j][ser_index] = last_idx.totalFalseTraced
                            accumulator[11][j][ser_index] = last_idx.totalFalseNegative
                            accumulator[12][j][ser_index] = last_idx.totalNonCompliant
                            
                    # add the inf-trace timelag histogram
                    timelag_hist.append(last_step.timelag_hist)
                    # positive detection and infection sources identified across the simulation
                    # without agent: detected is equal to true positives of tracing (and no distinction can be made between testing and tracing detections)
                    # with agent: detected is equal to last_step.totalDetected, variable which is incremented for each newly identified positive node through testing
                    detected.append(last_step.totalDetected if use_agent else last_step.totalTraced - last_step.totalFalseTraced)
                    test_src.append(last_step.testSrc)
                    test_src_unique.append(last_step.testSrcUnique)
                    trace_src.append(last_step.traceSrc)
                    trace_src_unique.append(last_step.traceSrcUnique)

                # this branch is needed because otherwise means and stds will be nan
                else:
                    for i in (detected, test_src, test_src_unique, trace_src, trace_src_unique):
                        i.append(0)
                
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
                        # The growth_rate represents new_cases / active_cases_past
                        growth.append(new_infections / reference_past_active_inf)
                        # The active_growth_rate represents active_cases_now / active_cases_past
                        active_growth.append(active_infected / reference_past_active_inf)
                        # change reference past measurement to use for the next Reff computation
                        reference_past_total_inf = total_infected
                        reference_past_time = event_time
                        reference_past_active_inf = active_infected

                i_max.append(this_i_max)
                i_timeofmax.append(this_i_timeofmax)
                h_max.append(this_h_max)
                h_timeofmax.append(this_h_timeofmax)
                # note growth_series and active_growth_series are lists of lists
                growth_series.append(growth if growth else [0])
                active_growth_series.append(active_growth if active_growth else [0])
                
            # indexes to remove from the alternative mean_wo and std_wo calculation (only if option selected from args)
            without_idx = idx_early_stopped if args.avg_without_earlystop else None
                        
            ###############
            # compute averages and other statistics for all simulation result parameters over time
            stats_for_timed_parameters = []
            stats_for_laststamp = []
            for i in range(num_vars):
                current_stats = accumulator[i]
                stats_for_timed_parameters.append(get_statistics(current_stats, compute=STATS_MEAN_OPT, axis=1, avg_without_idx=without_idx))
                stats_for_laststamp.append(get_statistics(current_stats[-1], compute=STATS_ALL_OPT, avg_without_idx=without_idx)[0])
                                
            # compute averages and other statistics for the peak simulation results
            stats_for_max_inf = get_statistics(i_max, compute=STATS_ALL_OPT, avg_without_idx=without_idx)[0]
            stats_for_timeofmax_inf = get_statistics(i_timeofmax, compute=STATS_MEAN_WO_OPT, avg_without_idx=without_idx)[0]
            stats_for_max_hos = get_statistics(h_max, compute=STATS_ALL_OPT, avg_without_idx=without_idx)[0]
            stats_for_timeofmax_hos = get_statistics(h_timeofmax, compute=STATS_MEAN_WO_OPT, avg_without_idx=without_idx)[0]
            stats_for_detected = get_statistics(detected, compute=STATS_MEAN_OPT, avg_without_idx=without_idx)[0]
            stats_for_test_src = get_statistics(test_src, compute=STATS_MEAN_OPT, avg_without_idx=without_idx)[0]
            stats_for_test_src_unique = get_statistics(test_src_unique, compute=STATS_MEAN_OPT, avg_without_idx=without_idx)[0]
            stats_for_trace_src = get_statistics(trace_src, compute=STATS_MEAN_OPT, avg_without_idx=without_idx)[0]
            stats_for_trace_src_unique = get_statistics(trace_src_unique, compute=STATS_MEAN_OPT, avg_without_idx=without_idx)[0]
            
            # compute averages and other stats for growth rates
            stats_for_growth = get_statistics(ut.pad_2d_list_variable_len(growth_series),
                                              compute=STATS_MEAN_WO_OPT, avg_without_idx=without_idx)
            stats_for_active_growth = get_statistics(ut.pad_2d_list_variable_len(active_growth_series),
                                                     compute=STATS_MEAN_WO_OPT, avg_without_idx=without_idx)
            
            # compute total inf-trace timelag histogram across all simulations
            total_timelag_hist = sum(timelag_hist, Counter())
            
            ##############
            # update averages dictionary for the current parameter value
            # the key for results in the summary dictionary will be 'res' if the simulations ran for one parameter value only
            # otherwise the key will be the actual parameter value
            key_for_res = 'res' if len(self.param_res) == 1 else param
            current = summary[key_for_res]
            current['time'] = time_range_limits
            current['early-stopped'] = len(idx_early_stopped)
            current['timelag-hist'] = total_timelag_hist
            
            current['average-infected'] = stats_for_timed_parameters[0]
            current['average-max-infected'] = stats_for_max_inf
            current['average-time-of-max-infected'] = stats_for_timeofmax_inf
            
            current['average-total-infected'] = stats_for_timed_parameters[1]
            current['average-overall-infected'] = stats_for_laststamp[1]
            
            overall_infected_fractions = accumulator[1][-1] / args.netsize
            # percentage of healthy nodes across simulations - healthy/netsize = 1 - overall_infected/netsize
            current['average-%healthy'] = get_statistics(1 - overall_infected_fractions,
                                                         compute=STATS_ALL_OPT, round_to=3,
                                                         avg_without_idx=without_idx)[0]
            # fraction of simulations (out of the total len_series) that finished with overall_infected less than a fraction alpha_contain
            current['average-%contained'] = round(np.count_nonzero(overall_infected_fractions < args.alpha_contain) / len_series, 2)
            
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
            overall_true_trace = Counter(current['average-overall-traced'])
            overall_true_trace.subtract(current['average-overall-false-traced'])
            current['average-overall-true-traced'] = dict(overall_true_trace)
            
            current['average-total-false-negative'] = stats_for_timed_parameters[11]
            current['average-overall-false-negative'] = stats_for_laststamp[11]
            
            current['average-total-noncompliant'] = stats_for_timed_parameters[12]
            current['average-overall-noncompliant'] = stats_for_laststamp[12]
            
            current['average-overall-detected'] = stats_for_detected 
            current['average-test-src'] = stats_for_test_src
            current['average-test-src-uniq'] = stats_for_test_src_unique
            current['average-trace-src'] = stats_for_trace_src
            current['average-trace-src-uniq'] = stats_for_trace_src_unique

            if args.gamma and args.miup and args.model == 'covid':
                # for COVID, total time of infectiousness: presymp + symp/asymp duration
                infectious_time_rate = 1 / (1/args.miup + 1/args.gamma)
            else:
                # in SIR and SEIR, the total infectious time is given by `gamma`
                infectious_time_rate = args.gamma
            # basic R0 is scaled by the average number of contacts (since the transmission rate is likewise scaled)
            current['r0'] = args.avg_deg * args.beta / (infectious_time_rate + 1e-16)
            # This is based on Tsimring and Huerta, 2002
            current['r-trace'] = args.avg_deg * args.beta / (infectious_time_rate + args.beta + args.taur *
                                    (1 + current['average-overall-true-traced'][STATS_MEAN_OPT] / current['average-overall-infected'][STATS_MEAN_OPT]))
            # growth rates based on r_window
            current['growth'] = stats_for_growth
            # for COVID models, we can also compute R_eff during the first period of the simulation from the growth recorded within the same period
            # since we know both the mean and shape of the generation time ditribution
            if args.model == 'covid':
                # we want R to reflect the r_window chosen and be corresponding to the first estimate of the growth
                current['r-eff'] = ut.r_from_growth(stats_for_growth[0][STATS_MEAN_OPT], method='exp', t=r_window, mean=6.6, shape=1.87, inv_scale=0.28)
            current['active-growth'] = stats_for_active_growth

        if printit:
            # allow for args not to be printed in the summary if printit is a float and its decimal part is 5
            if printit * 10 % 10 == 5:
                del summary['args']
            else:
                # if nettype is a predefined supplied network (i.e. not a str), 
                # the repr of the net in the summary will be overwritten to reduce the output space
                if not isinstance(summary['args']['nettype'], str):
                    summary['args']['nettype'] = 'predefined'
            # write the summary to stdout or to a file
            if printit < 2:
                print(json.dumps(summary, indent=1, cls=ut.NumpyEncoder))
            else:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                # Output to a JSON file
                with open(args.save_path+'result.json', 'w') as f:
                    json.dump(summary, f, indent=1, cls=ut.NumpyEncoder)

        return summary