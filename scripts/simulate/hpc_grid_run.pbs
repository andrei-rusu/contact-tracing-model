#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#PBS -l nodes=1:ppn=16
#PBS -l walltime=00:45:00
#PBS -t 1
#PBS -e tracing_runs/job_output/pbs/error
#PBS -o tracing_runs/job_output/pbs/output

# Load the conda module and activate the environment if in PBS environment
if [ -n "$PBS_JOBID" ]; then
    echo "Running in PBS with job ID: $PBS_JOBID"
    module load conda
    source activate contact
else
    echo "Running locally, default PBS job ID to 0"
    PBS_JOBID=0
    PBS_O_WORKDIR=.
fi
#Change to directory from which job was submitted
cd $PBS_O_WORKDIR

# Set an ID for the run, if this wasn't run from a PBS array job
if [ -z "$PBS_ARRAYID" ]; then
    PBS_ARRAYID=0
fi
# set the job and task ID to the array ID
JOB_ID=$PBS_JOBID
TASK_ID=$PBS_ARRAYID

#Get a specific parameter configuration based on the ARRAYID using the paramgrid.py script
gridentry=($(python ../paramgrid.py --id $TASK_ID))
uptake=${gridentry[0]}
taut=${gridentry[1]}
taur=${gridentry[2]}
pa=${gridentry[3]}
overlap=${gridentry[4]}
group=${gridentry[5]}
dual=${gridentry[6]}

if [ $taut -eq 10 ]
then
    taut=(.05 .1 .2 .5)
fi

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg

python -m ct_simulator.run_tracing \
    --exp-id "pbs_grid_"$JOB_ID"" \
    --netsize 1000 \
    --k 10 \
    --p .2 \
    --nettype "powerlaw-cluster" \
    --multip 3 \
    --model "covid" \
    --dual $dual \
    --uptake $uptake \
    --overlap 1 \
    --maintain_overlap False \
    --overlap_two $overlap \
    --nnets 2 \
    --niters 2 \
    --separate_traced True \
    --avg_without_earlystop True \
    --trace_after 1 \
    --first_inf .1 \
    --earlystop_margin 0 \
    --rem_orphans True \
    --noncomp .001 \
    --noncomp_after 14 \
    --presample 100000 \
    --pa $pa \
    --taut ${taut[@]} \
    --delay_two 2. \
    --taur $taur \
    --sampling_type "min" \
    --seed $TASK_ID \
    --netseed 11 \
    --animate 0 \
    --summary_print 3