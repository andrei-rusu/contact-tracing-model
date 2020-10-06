import lib.network as net
import argparse

def main(args):
    
    trueNet = net.get_random(n=20, k=10)
    trueNet.change_state(9, 'I')
    trueNet.change_state(2, 'Is')
    
    knowNet = net.get_dual(trueNet, overlap=.8, z_add=0, z_rem=5)
    knowNet.change_traced_state_fast_update(9)
    
    print(knowNet.node_counts[args.nid])

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--nid', type=int, default=0)

    args = argparser.parse_args()

    main(args)