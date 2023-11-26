#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Grid Simulation on SocialEvolution"
#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:15:00          # walltime
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

#Get a specific parameter configuration based on the ARRAY_ID using the paramgrid.py script
gridentry=($(python ../paramgrid.py --id $SLURM_ARRAY_TASK_ID))
uptake=${gridentry[0]}
taut=${gridentry[1]}
taur=${gridentry[2]}
pa=${gridentry[3]}
overlap=${gridentry[4]}
group=${gridentry[5]}
dual=${gridentry[6]}

# circumvent normal logic for taut if a value of 10 has been supplied -> check for 4 different values for taut
if [ $taut -eq 10 ]
then
    taut=(0 .1 .2 .5 1.0 1.5 2.0)
fi

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg

python -m ct_simulator.run_tracing \
    --exp-id "socialevol_"$JOB_ID"" \
    --nettype "socialevol::40,70,w" \
    --use_weights True \
    --K_factor 10 \
    --update_after 1 \
    --multip 3 \
    --model "covid" \
    --dual $dual \
    --uptake $uptake \
    --overlap 1 \
    --maintain_overlap False \
    --overlap_two $overlap \
    --maintain_overlap_two True \
    --nnets 10 \
    --niters 150 \
    --separate_traced True \
    --avg_without_earlystop True \
    --trace_after 3 \
    --first_inf .1 \
    --earlystop_margin 5 \
    --rem_orphans False \
    --noncomp .001 \
    --noncomp_after 14 \
    --presample 100000 \
    --pa $pa \
    --taut ${taut[@]} \
    --delay_two 2. \
    --taur $taur \
    --sampling_type "min" \
    --infseed -1 \
    --seed $TASK_ID \
    --netseed 11 \
    --animate 0 \
    --summary_print 3