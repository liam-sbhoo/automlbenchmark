#!/bin/bash
# Usage: sbatch run_benchmark_slurm.sh <framework> <benchmark>


#SBATCH --job-name=time_series_benchmark
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gres=gpu:1                          # Number of GPUs
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task
#SBATCH --mem=32G                             # Total memory limit
#SBATCH --output=slurm_out/job_output_%j.txt  # Standard output and error log (%j expands to jobId)

# Load any modules or source your environment here if necessary
#module load python
source ~/.time_bashrc

# Run the experiment script with the SLURM_ARRAY_TASK_ID as an argument
AUTOMLBENCHMARK_CONFIG_PATH="$HOME/.config/automlbenchmark/benchmarks"
ID_TO_TASK_MAPPING_PATH=$AUTOMLBENCHMARK_CONFIG_PATH/"id_to_task_mapping.yaml"

FRAMEWORK=$1
BENCHMARK=$2
TASK_NAME=$3
CONSTRAINT="4h16c"

WANDB_PROJECT="tabpfn-time-series"

echo "Running benchmark on task $TASK_NAME"
python runbenchmark.py $FRAMEWORK $BENCHMARK $CONSTRAINT \
  -t $TASK_NAME \
  --wandb_project $WANDB_PROJECT \
  --wandb_tags $CONSTRAINT \
  --debug_jid $SLURM_JOB_ID
