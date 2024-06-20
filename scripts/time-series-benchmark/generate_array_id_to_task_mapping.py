from pathlib import Path
import yaml
import argparse

from generate_task_configs import dataset_metadata


ID_2_TASK_FILENAME = "id_to_task_mapping.yaml"


def main(args):
    # Load benchmark definition and create a mapping of ID to task names
    with open(args.benchmark_path, "r") as f:
        benchmark_definition: list = yaml.safe_load(f)
    id_2_task = {i: task['name'] for i, task in enumerate(benchmark_definition)}

    # Write the mapping to a YAML file
    output_file = Path(args.output_path) / ID_2_TASK_FILENAME
    print(f"Writing ID_2_TASK mapping into {output_file}")
    with open(output_file, "w") as out_file:
        yaml.safe_dump(id_2_task, out_file)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_path", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    main(args)
