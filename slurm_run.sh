#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Simulation"
#SBATCH --ntasks-per-node=8     # Tasks per node
###SBATCH --ntasks=1               # Number of total tasks
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:30:00          # walltime
#SBATCH -o data/run/job_output/slurm/slurm-%A_%a.out        # STDOUT
#SBATCH -e data/run/job_output/slurm/slurm-%A_%a.err        # STDERR

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    SLURM_ARRAY_TASK_ID=0
fi

#Change to directory from which job was submitted
cd "$HOME/contact-tracing-model"

NETSIZE=200

# newfile="data/run/batch2_slurm/simresult"$SLURM_ARRAY_TASK_ID".json"
newfile="data/run/batch2_slurm/simresult_"$NETSIZE".json"

module load conda/py3-latest
source activate contact

python run.py \
    --netsize $NETSIZE \
    --k 10 \
    --multip 3 \
    --model 'covid' \
    --dual 1 \
    --overlap 1 \
    --uptake .5 \
    --maintain_overlap False \
    --nnets 4 \
    --niters 4 \
    --separate_traced True \
    --noncomp 0 \
    --taut .1 \
    --taur .1 > $newfile

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile