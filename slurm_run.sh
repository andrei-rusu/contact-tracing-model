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

newfile="data/run/batch2_slurm/simresult"$SLURM_ARRAY_TASK_ID".json"

module load conda/py3-latest
source activate contact

python run.py \
    --netsize 100 \
    --k 10 \
    --multip False \
    --model 'covid' \
    --dual True \
    --overlap .8 \
    --nnets 1 \
    --niters 1 \
    --separate_traced True \
    --seed $SLURM_ARRAY_TASK_ID \
    --taut .1 \
    --taur .005 > $newfile

# Remove running artefacts from sim results
sed -i -n '/args/,$p' $newfile