#!/bin/bash
# Job script to run slurm job
#=============================================
#SBATCH --job-name="Epidemic Simulation"
#SBATCH --ntasks-per-node=20     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=02:30:00          # walltime
#SBATCH -o tracing_runs/job_output/slurm/slurm-%A_%a.out
#SBATCH -e tracing_runs/job_output/slurm/slurm-%A_%a.err

# Load the conda module and activate the environment if in SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in SLURM with job ID: $SLURM_JOB_ID"
    module load conda
    source activate contact
else
    echo "Running locally, default SLURM job ID to 0"
    SLURM_JOB_ID=0
    SLURM_SUBMIT_DIR=.
fi
# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Set an ID for the run, if this wasn't run from a SLURM array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi
# set the job and task ID to the array task ID
JOB_ID=$SLURM_JOB_ID
TASK_ID=$SLURM_ARRAY_TASK_ID

netsize=200

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg
    
python -m ct_simulator.run_tracing \
    --exp-id "slurm_"$netsize"" \
    --netsize $netsize \
    --k 10 \
    --p .2 \
    --nettype "random" \
    --multip 3 \
    --model "covid" \
    --dual 1 \
    --uptake .5 \
    --overlap 1 \
    --maintain_overlap False \
    --nnets 10 \
    --niters 15 \
    --separate_traced True \
    --avg_without_earlystop True \
    --trace_after 1 \
    --first_inf 1 \
    --earlystop_margin 2 \
    --rem_orphans True \
    --noncomp 0 \
    --presample 100000 \
    --pa .2 \
    --taut .1 \
    --taur .1 \
    --sampling_type "min" \
    --seed $TASK_ID \
    --netseed 11 \
    --animate 0 \
    --summary_print 3