#!/usr/bin/env python3

import argparse
import os
import yaml
import submitit
from pathlib import Path
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and run sbatch script for time series benchmark")
    parser.add_argument("--framework", required=True, help="Framework to use")
    parser.add_argument("--benchmark", required=True, help="Benchmark to run")
    parser.add_argument("--cluster_partition", default="mlhiwidlc_gpu-rtx2080", help="Cluster partition to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--task", default="all", help="Task to run, either a single task or 'all'")
    parser.add_argument("--constraint", default="4h16c", help="Constraint for the job")
    return parser.parse_args()

def retrieve_task_names_from_benchmark_def(benchmark_def_file):
    with open(benchmark_def_file, 'r') as f:
        benchmark_def = yaml.safe_load(f)

    task_names = [task_def["name"] for task_def in benchmark_def]
    return task_names

def main():
    args = parse_arguments()
    job_name = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory = 32
    num_gpus = 1
    num_cpus = 16   # hard-coded for now (ignoring the constraints)

    benchmark_script_path = Path(__file__).parent.parent.parent / "runbenchmark.py"
    
    # Retrieve the task map from the benchmark definition
    automlbenchmark_config_path = Path.home() / ".config" / "automlbenchmark" / "benchmarks"
    benchmark_def_file = automlbenchmark_config_path / f"{args.benchmark}.yaml"
    all_tasks = retrieve_task_names_from_benchmark_def(benchmark_def_file)
    
    # If a single task is specified, check if it is valid
    if args.task != "all":
        if args.task not in all_tasks:
            raise ValueError(f"Task {args.task} not found in benchmark definition")
        tasks = [args.task]
    else:
        tasks = all_tasks

    num_tasks = len(tasks)

    # Report the benchmark parameters
    print("\nRunning benchmark with the following parameters:")
    print(f" . FRAMEWORK: {args.framework}")
    print(f" . BENCHMARK: {args.benchmark}")
    print(f" . CLUSTER_PARTITION: {args.cluster_partition}")
    print(f" . SEED: {args.seed}")
    print(f" . TASK: {args.task}")
    print(f" . CONSTRAINT: {args.constraint}")
    print(f" . # TASKS: {num_tasks}")

    # Setup submitit executor
    executor = submitit.AutoExecutor(folder=f"slurm_logs/{job_name}")
    executor.update_parameters(
        name=job_name,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=num_cpus,
        mem_gb=memory,
        slurm_gres=f"gpu:{num_gpus}",
        slurm_partition=args.cluster_partition,
        slurm_array_parallelism=num_tasks,
        slurm_setup=["source ~/.time_bashrc"],
        timeout_min=1439  # 23 hours and 59 minutes
    )

    jobs = []
    with executor.batch():
        for task_name in tasks:

            cmd = ["python", str(benchmark_script_path)]
            script_args = [
                args.framework,
                args.benchmark,
                args.constraint,
            ]
            script_kwargs = {
                "task": task_name,
                "wandb_project": "tabpfn-time-series",
                "wandb_tags": args.constraint,
                "wandb_group_id": job_name,
                "seed": args.seed
            }

            job = executor.submit(submitit.helpers.CommandFunction(cmd), *script_args, **script_kwargs)
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")
    
    # # Wait for all jobs to complete
    # for job in jobs:
    #     job.result()

if __name__ == "__main__":
    main()
