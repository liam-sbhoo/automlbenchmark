#!/usr/bin/env python3

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run time series benchmark on a single GPU")
    parser.add_argument("--framework", required=True, help="Framework to use")
    parser.add_argument("--benchmark", required=True, help="Benchmark to run")
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
    print(f" . SEED: {args.seed}")
    print(f" . TASK: {args.task}")
    print(f" . CONSTRAINT: {args.constraint}")
    print(f" . # TASKS: {num_tasks}")

    for task_name in tasks:
        cmd = [
            "python", str(benchmark_script_path),
            args.framework,
            args.benchmark,
            args.constraint,
            "--task", task_name,
            "--wandb_project", "tabpfn-time-series",
            "--wandb_tags", " ".join([args.constraint, args.framework, "gpu-v100"]),
            "--seed", str(args.seed)
        ]

        print(f"\nRunning task: {task_name}")
        subprocess.run(cmd, check=True)

    print(f"\nCompleted {num_tasks} tasks")

if __name__ == "__main__":
    main()
