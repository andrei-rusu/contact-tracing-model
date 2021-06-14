#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Grid Simulation"
#SBATCH --ntasks-per-node=4     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:10:00          # walltime
#SBATCH -o data/run/job_output/slurm/slurm-%A_%a.out        # STDOUT
#SBATCH -e data/run/job_output/slurm/slurm-%A_%a.err        # STDERR

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    SLURM_ARRAY_TASK_ID=0
fi

#Change to directory from which job was submitted
cd "$HOME/contact-tracing-model"

module load conda/py3-latest
source activate contact

#Get a specific parameter configuration based on the ARRAY_ID using the gridsearch.py script
gridentry=($(python lib/gridsearch.py --id $SLURM_ARRAY_TASK_ID))
#The entries are:
# taut (contact tracing rate); taur (random tracing / testing rate); 
# overlap (app uptake); pa (prob of being asymptomatic)
uptake=${gridentry[0]}
taut=${gridentry[1]}
taur=${gridentry[2]}
pa=${gridentry[3]}
overlap=${gridentry[4]}
dual=${gridentry[5]}

# circumvent normal logic for taut if a value of 10 has been supplied -> check for 4 different values for taut
if [ $taut -eq 10 ]
then
    taut=(.05 .1 .2 .5)
fi

newfile="data/run/social_slurm/simresult_id"$SLURM_ARRAY_TASK_ID"_"$taut"_"$taur"_"$uptake".json"

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg

python run.py \
    --nettype "socialevol" \
    --use_weights True \
    --K_factor 10 \
    --multip 2 \
    --model "covid" \
    --dual $dual \
    --uptake $uptake \
    --overlap 1 \
    --maintain_overlap False \
    --overlap_two $overlap \
    --maintain_overlap_two True \
    --nnets 1 \
    --niters 1000 \
    --separate_traced True \
    --avg_without_earlystop True \
    --trace_after 7 \
    --first_inf .1 \
    --earlystop_margin 0 \
    --rem_orphans False \
    --noncomp .001 \
    --noncomp_after 14 \
    --presample 100000 \
    --pa $pa \
    --taut ${taut[@]} \
    --delay_two 2. \
    --taur $taur \
    --sampling_type "min" \
    --netseed 31 \
    --seed 11 > $newfile 

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile