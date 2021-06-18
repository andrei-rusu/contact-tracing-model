import pandas as pd
import numpy as np


def load_socialevol(agg='D', proxim_prob=None):
    call = pd.read_csv('data/SocialEvolution/Calls.csv.bz2')
    sms = pd.read_csv('data/SocialEvolution/SMS.csv.bz2', encoding="ISO-8859-1")
    proxim = pd.read_csv('data/SocialEvolution/Proximity.csv.bz2')
#     wlan = pd.read_csv('data/SocialEvolution/WLAN2.csv.bz2')
    # Deal with naming issues
    uid = 'uid'
    dest = 'dst'
    date = 'time'
    ct = 'counts'
    week_idx = 'weekid'
    time_idx = 'timeid'
    sms.rename(columns={'user.id': uid, 'dest.user.id.if.known': dest}, inplace=True)
    call.rename(columns={'user_id': uid, 'dest_user_id_if_known': dest, 'time_stamp': date}, inplace=True)
    proxim.rename(columns={'user.id': uid, 'remote.user.id.if.known': dest}, inplace=True)
    
    # filter NaNs in id, and convert nids to int and date to datetime64[D]
    sms_filt = sms[sms[uid].notnull() & sms[dest].notnull()].loc[sms[uid] != sms[dest]]
    call_filt = call[call[uid].notnull() & call[dest].notnull()].loc[call[uid] != call[dest]]
    # we select only the proximity entries that have a higher chance of having been on the same floor
    proxim_filt = proxim[proxim.prob2 >= proxim_prob] if proxim_prob is not None else proxim
    
    cols = [time_idx, date, week_idx, uid, dest]
    
    def get_counts_by_time_agg(df):
        df = df.astype({uid: 'int', dest: 'int', date: 'datetime64[D]'}, copy=False)
        year = df[date].dt.isocalendar().year
        # we only consider dates which span weeks from 2008 and 2009
        df = df[year.isin((2008, 2009))]
        # we add by default a column specifying the week id (start index 0) of the datapoints for filtering purposes
        df[week_idx] = (year - 2008) * 52 + (df[date].dt.isocalendar().week - 1)
        # based on the value chosen for agg, we'll have either the week id or the day id as a time id (start index 0)
        df[time_idx] = df[week_idx] if agg == 'W' else (year - 2008) * 366 + (df[date].dt.dayofyear - 1)
        # eliminate different orders for the 2 nids
        df[[uid, dest]] = np.sort(df[[uid, dest]], axis=1)
        # group by timeid, and node ids
        df = df.groupby(cols).size().reset_index(name=ct)
        return df
    

    sms_filt = get_counts_by_time_agg(sms_filt)
    call_filt = get_counts_by_time_agg(call_filt)
    proxim_filt = get_counts_by_time_agg(proxim_filt)
        
    # outer join of sms and call counts, and sum all counts over based on week, source, destination
    merge_sms_call = pd.merge(sms_filt, call_filt, on=cols, how='outer').set_index(cols).sum(axis=1).reset_index(name=ct).astype({ct:'int'})
    
    return (proxim_filt, merge_sms_call)
    
    
class DataLoader():
    
    def __init__(self, dataset='socialevol', agg='D', proxim_prob=None):
        self.dataset = dataset
        self.agg = agg
        # load data as specified by the dataset and agg args
        self.data = globals()['load_' + dataset](agg, proxim_prob)
        
    def get_edge_data_for_time(self, which_inf=0, which_tr=1, time_fr='2009-01-05', time_to='2009-06-14', use_week_index=True):
        """
        date_fr/to can either be strings marking dates or integers marking the time ids (day or week ids) directly
        """
        pd.options.mode.chained_assignment = None
        if which_inf is None or which_inf < 0:
            raise ValueError('A valid network must be selected for the infection net')
        vars_to_query = ['uid', 'dst', 'counts']
        inf = self.data[which_inf]
        # filtering data can be done based on actual datetime (if time_fr is in str format), by weekid (if use_week_index=True), or by timeid otherwise
        if isinstance(time_fr, str):
            time_selector = 'time'
        elif use_week_index:
            time_selector = 'weekid'
        else:
            time_selector = 'timeid'
        # select data inbetween these dates/time_ids based on time_selector chosen
        inf = inf.loc[inf[time_selector].between(time_fr, time_to)]
        # make first time id point 0, and amend the other entries to reflect this reindexing
        offset = inf.timeid.min()
        inf.timeid -= offset
        # get the time indexes
        time_keys = set(inf.timeid)
        nodes = set(inf.uid).union(inf.dst)
        # we do not dilute the edge weights if there are holes in the time_keys -> assume data collection problem rather than no interaction
        norm_for_avg = (inf.time.max() - inf.time.min()).days * len(nodes)
        if which_tr is not None and which_tr >= 0:
            tr = self.data[which_tr]
            # filter tracing net data to include only nodes available in the infection net and dates inbetween what is specified
            tr = tr.loc[tr.uid.isin(nodes) & tr.dst.isin(nodes) & tr.time.between(date_fr, date_to)]
            # apply offset inferred from infection net
            tr.timeid -= offset
            # filter out data in the tracing network that was before the first data in the infection net
            tr = tr[tr.timeid >= 0]
            # union of the set of all time index keys
            time_keys = time_keys.union(tr.timeid)
            # format for return is {time_index : (infection network edges, tracing network edges)}
            edges = {str(int(timeid)): (inf.loc[inf.timeid == timeid][vars_to_query].values,
                                        tr.loc[tr.timeid == timeid][vars_to_query].values) for timeid in time_keys}
            # normalization <W> for tracing net - average over timesteps of averages over nodes
            edges['Wt'] = tr.counts.sum() / norm_for_avg
        else:
            edges = {str(int(timeid)): (inf.loc[inf.timeid == timeid][vars_to_query].values, None) for timeid in time_keys}
        # normalization <W> for infection net - average over timesteps of averages over nodes
        sum_all = inf.counts.sum()
        edges['Wi'] = 2 * sum_all / norm_for_avg
        # we also set a key for retrieving node ids in this data
        edges['nid'] = nodes
        return edges