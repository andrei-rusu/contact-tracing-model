#!/bin/bash
# Job script to run pbs job in job array
#=============================================
# set default resource requirements for job 
# - these can be overridden on the qsub command line
#
#SBATCH --job-name="Epidemic Simulation"
###SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --ntasks=1               # Number of total tasks
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

NETSIZE=200

# newfile="data/run/batch2_slurm/simresult"$SLURM_ARRAY_TASK_ID".json"
newfile="data/run/batch2_slurm/simresult_"$NETSIZE".json"

module load conda/py3-latest
source activate contact

python run.py \
    --netsize $NETSIZE \
    --k 10 \
    --multip True \
    --model 'covid' \
    --dual 1 \
    --overlap 1 \
    --uptake .5 \
    --maintain_overlap False \
    --nnets 3 \
    --niters 3 \
    --separate_traced True \
    --taut .01 \
    --taur .01 > $newfile

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile