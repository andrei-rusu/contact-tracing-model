python -m ct_simulator.run_tracing -a config/agent_config.json -r config/amodel_config.json ^
    --exp_id "win_agent/" ^
    --nettype "barabasi:5:1" ^
    --netsize 100 ^
    --k 2.8 ^
    --p .1 ^
    --rem_orphans False ^
    --use_weights True ^
    --dual 0 ^
    --edge_sample_size 0.4 0.4 ^
    --control_schedule 3 0.5 0.99 ^
    --control_after 5 ^
    --control_after_inf .05 ^
    --control_initial_known .25 ^
    --control_gpu 0 ^
    --first_inf 5 ^
    --taut 0 ^
    --taur 0 ^
    --sampling_type "min" ^
    --presample 10000 ^
    --model "seir" ^
    --spontan False ^
    --pa=.2 ^
    --update_after 1 ^
    --summary_splits 20 ^
    --noncomp 0 ^
    --noncomp_after 10000 ^
    --avg_without_earlystop True ^
    --seed 32 ^
    --netseed 25 ^
    --infseed -1 ^
    --multip 0 ^
    --nnets 1 ^
    --niters 1 ^
    --animate 0 ^
    --summary_print 3
