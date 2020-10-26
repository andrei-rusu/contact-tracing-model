#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Grid Simulation"
#SBATCH --ntasks-per-node=20     # Tasks per node
###SBATCH --ntasks=1               # Number of total tasks
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:30:00          # walltime
#SBATCH -o data/run/job_output/slurm/slurm.out        # STDOUT
#SBATCH -e data/run/job_output/slurm/slurm.err        # STDERR

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
taut=${gridentry[0]}
taur=${gridentry[1]}
pa=${gridentry[2]}
overlap=${gridentry[3]}

newfile="data/run/batch2_slurm/simresult_id"$SLURM_ARRAY_TASK_ID"_"$taut"_"$taur"_"$overlap".json"

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg

python run.py \
    --netsize 1000 \
    --k 10 \
    --multip True \
    --model "covid" \
    --dual True \
    --overlap $overlap \
    --nnets 5 \
    --niters 20 \
    --separate_traced True \
    --noncomp .02 \
    --pa $pa \
    --taut $taut \
    --taur $taur > $newfile

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile