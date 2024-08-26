#!/bin/bash

# This file is used to generate sbatch script (depending on the input) and run the script
# Usage: ./run_benchmark_slurm_master.sh --framework FRAMEWORK --benchmark BENCHMARK [--cluster_partition CLUSTER_PARTITION] [--seed SEED]

source ~/.time_bashrc > /dev/null 2>&1

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift ;;
        --benchmark) BENCHMARK="$2"; shift ;;
        --cluster_partition) CLUSTER_PARTITION="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --constraint) CONSTRAINT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# assert FRAMEWORK and BENCHMARK are not empty
if [ -z "$FRAMEWORK" ] || [ -z "$BENCHMARK" ]; then
    echo "ERROR: --framework and --benchmark must be provided"
    echo "Usage: ./run_benchmark_slurm_master.sh --framework FRAMEWORK --benchmark BENCHMARK [--cluster_partition CLUSTER_PARTITION] [--seed SEED]"
    exit 1
fi

# if CLUSTER_PARTITION is not provided, set to default value (mlhiwidlc_gpu-rtx2080)
if [ -z "$CLUSTER_PARTITION" ]; then
    CLUSTER_PARTITION="mlhiwidlc_gpu-rtx2080"
fi

# if SEED is not provided, set to default value (0)
if [ -z "$SEED" ]; then
    SEED=0
fi

# if TASK is not provided, set to default value (all tasks)
if [ -z "$TASK" ]; then
    TASK="all"
fi

# if CONSTRAINT is not provided, set to default value (4h16c)
if [ -z "$CONSTRAINT" ]; then
    CONSTRAINT="4h16c"
fi

echo ""
echo "Running benchmark with the following parameters:"
echo " . FRAMEWORK: $FRAMEWORK"
echo " . BENCHMARK: $BENCHMARK"
echo " . CLUSTER_PARTITION: $CLUSTER_PARTITION"
echo " . SEED: $SEED"
echo " . TASK: $TASK"
echo " . CONSTRAINT: $CONSTRAINT"

JOB_NAME=time_series_benchmark
MEMORY=32G
NUM_GPUS=1
NUM_CPUS=16

# Count the number of task
AUTOMLBENCHMARK_CONFIG_PATH=$HOME/.config/automlbenchmark/benchmarks
ID_TO_TASK_MAPPING_PATH=$AUTOMLBENCHMARK_CONFIG_PATH/id_to_task_mapping.yaml

NUM_TASKS=$(python <<EOF
import yaml
with open('$ID_TO_TASK_MAPPING_PATH', 'r') as f:
    id_to_task_mapping = yaml.safe_load(f)
    print(len(id_to_task_mapping))
EOF
)
echo "Number of tasks: $NUM_TASKS"

ARRAY_TASKS="0-$((NUM_TASKS - 1))"
echo "Array task: $ARRAY_TASKS"

# Generate sbatch script
CURR_FILE_DIR=$(dirname $(realpath $0))
SBATCH_SCRIPT_DIR=$CURR_FILE_DIR/generated
SBATCH_SCRIPT_NAME=run_benchmark_slurm.sh
mkdir -p $SBATCH_SCRIPT_DIR
SBATCH_SCRIPT_PATH=$SBATCH_SCRIPT_DIR/$SBATCH_SCRIPT_NAME

# Create the sbatch script
cat <<EOT > $SBATCH_SCRIPT_PATH
#!/bin/bash
# Usage: sbatch run_benchmark_slurm.sh <framework> <benchmark>

#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gres=gpu:$NUM_GPUS                  # Number of GPUs
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --cpus-per-task=$NUM_CPUS             # Number of CPU cores per task
#SBATCH --mem=$MEMORY                         # Total memory limit
#SBATCH --array=$ARRAY_TASKS                  # Total number of datasets in the benchmark
#SBATCH --output=slurm_out/job_output_%j.txt  # Standard output and error log (%j expands to jobId)

# Load any modules or source your environment here if necessary
source ~/.time_bashrc

# Run the experiment script with the SLURM_ARRAY_TASK_ID as an argument
AUTOMLBENCHMARK_CONFIG_PATH=$HOME/.config/automlbenchmark/benchmarks
ID_TO_TASK_MAPPING_PATH=$AUTOMLBENCHMARK_CONFIG_PATH/id_to_task_mapping.yaml

FRAMEWORK=$FRAMEWORK
BENCHMARK=$BENCHMARK
CONSTRAINT=$CONSTRAINT
SEED=$SEED

TASK_NAME=\$(python <<EOF
import yaml
with open('\$ID_TO_TASK_MAPPING_PATH', 'r') as file:
    tasks = yaml.safe_load(file)
    print(tasks[int(\$SLURM_ARRAY_TASK_ID)])
EOF
)
WANDB_PROJECT="tabpfn-time-series"

echo "Running benchmark on task \$TASK_NAME"
python runbenchmark.py \$FRAMEWORK \$BENCHMARK \$CONSTRAINT \\
  -t \$TASK_NAME \\
  --wandb_project \$WANDB_PROJECT \\
  --wandb_tags \$CONSTRAINT \\
  --wandb_group_id \$SLURM_ARRAY_JOB_ID \\
  --debug_jid \$SLURM_JOB_ID \\
  --seed \$SEED
EOT

echo "Generated sbatch script: $SBATCH_SCRIPT_PATH"

# Submit the sbatch script
echo "Running sbatch script..."
sbatch -p $CLUSTER_PARTITION $SBATCH_SCRIPT_PATH
