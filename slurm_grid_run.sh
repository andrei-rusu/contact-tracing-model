#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Grid Simulation"
#SBATCH --ntasks-per-node=20     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=01:00:00          # walltime
#SBATCH -o data/run/job_output/slurm/slurm-%A_$a.out        # STDOUT
#SBATCH -e data/run/job_output/slurm/slurm-%A_$a.err        # STDERR

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

newfile="data/run/batch3_slurm/simresult_id"$SLURM_ARRAY_TASK_ID"_"$taut"_"$taur"_"$uptake".json"

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg

python run.py \
    --netsize 1000 \
    --k 10 \
    --multip 3 \
    --model 'covid' \
    --dual 1 \
    --overlap $overlap \
    --uptake $uptake \
    --maintain_overlap False \
    --nnets 30 \
    --niters 20 \
    --separate_traced True \
    --avg_without_earlystop True \
    --first_inf 1 \
    --earlystop_margin 4 \
    --rem_orphans True \
    --noncomp .002 \
    --presample 500 \
    --pa $pa \
    --taut $taut \
    --taur $taur > $newfile

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile