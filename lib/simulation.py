from lib.utils import Event

class SimEvent(Event):
    pass

class Simulation():
    
    def __init__(self, net, trans):
        self.net = net
        self.trans = trans
        self.time = 0
        
    def get_next_event(self):
        id_next = None
        from_next = None
        to_next = None
        best_time = float('inf')
        for nid, current_state in enumerate(self.net.node_states):
            for to, rate_func in self.trans[current_state]:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = rate_func(self.net, nid)
                # increment time with the current rate
                trans_time = self.time + rate
                if trans_time < best_time:
                    best_time = trans_time
                    from_next = current_state
                    to_next = to
                    id_next = nid
        if id_next is None:
            return None
        return SimEvent(node=id_next, fr=from_next, to=to_next, time=best_time)
                
        
    def run_event(self, e):
        self.net.change_state_fast_update(e.node, e.to)
        self.time = e.time
        
    def run_until(self, time_limit):
        if self.time >= time_limit:
            raise ValueError("WARNING: run_until called with a passed time")
        e = get_next_event()
        if e.time < time_limit:
            run_event(e)
        
        
    
        