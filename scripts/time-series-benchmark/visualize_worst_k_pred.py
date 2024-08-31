from argparse import ArgumentParser
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tabpfn_ts.data import TimeSeriesDataFrame


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRAMEWORK_DEF_FILE = REPO_ROOT / "resources" / "frameworks.yaml"
TABPFN_TS_CONFIG_FILE_DIR = Path("/home/hoos/hoos-time/playground/tabpfn-time-series/tabpfn_ts/pipeline/config")
FORECAST_DATASET_CONFIG_FILE = REPO_ROOT / "resources" / "benchmarks" / "point_forecast.yaml"

import sys

# Add the repository root to the Python path
sys.path.append(str(REPO_ROOT))


def get_pipeline(pipeline_name: str):
    from tabpfn_ts.pipeline.factory import PipelineFactory

    with open(FRAMEWORK_DEF_FILE, "r") as f:
        framework_defs = yaml.safe_load(f)

    # Convert all keys in framework_defs to lowercase
    framework_defs = {k.lower(): v for k, v in framework_defs.items()}

    pipeline_config_file = TABPFN_TS_CONFIG_FILE_DIR / framework_defs[pipeline_name]["params"]["config_file"]
    return PipelineFactory.from_config_file(pipeline_config_file)


def extract_history(history_dir: Path):
    history_pred_dir = history_dir / "predictions"
    
    # List all entries in the history_pred_dir
    entries = list(history_pred_dir.glob('*'))

    # Assert that there's only one entry and it's a directory
    assert len(entries) == 1, f"Expected exactly one entry in {history_pred_dir}, found {len(entries)}"
    assert entries[0].is_dir(), f"Expected a directory in {history_pred_dir}, found a file"
    entry = entries[0]

    task_name = entry.name

    # Assert only 1 seed is found
    seeds = list(entry.glob('*'))
    assert len(seeds) == 1, f"Expected exactly one seed in {entry}, found {len(seeds)}"
    seed = seeds[0]

    predictions_history = pd.read_csv(seed / "predictions.csv")
    
    return {
        "task_name": task_name,
        "predictions_history": predictions_history,
    }
    

def compute_individual_mase(df: pd.DataFrame):
    return ((df["truth"] - df["predictions"]).abs() / df["repeated_abs_seasonal_error"]).mean()


def save_mase_dist_plot(df_mase: pd.Series, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the distribution of MASE
    sns.histplot(df_mase, kde=True, ax=ax)

    # Set labels and title
    ax.set_xlabel('MASE')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Absolute Scaled Error (MASE)')

    # Add a vertical line for the mean MASE
    mean_mase = df_mase.mean()
    ax.axvline(mean_mase, color='r', linestyle='--', label=f'Mean MASE: {mean_mase:.2f}')

    # Add legend
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_task_config(task_name: str):
    with open(FORECAST_DATASET_CONFIG_FILE, "r") as f:
        raw_config = yaml.safe_load(f)

    task_config = {}
    for c in raw_config:
        if c["name"] == task_name:
            task_config = c
            break

    assert task_config is not None, f"Task {task_name} not found in the dataset config"

    return task_config


def get_task_dataset(task_name: str):
    
    from amlb.utils.core import Namespace
    from amlb.datasets.file import TimeSeriesDataset
    from frameworks.shared.utils import load_timeseries_dataset

    task_config = get_task_config(task_name)
    dataset_config = task_config["dataset"]

    print(f"Loading {task_name}")
    
    dataset = TimeSeriesDataset(
        path=dataset_config["path"],
        fold=0,
        target=dataset_config["target"],
        features="timestamp",
        cache_dir="./tmp_cache",
        config=Namespace(dataset_config | {"name": task_name})
    )
    dataset.train_path = dataset.train.path
    dataset.test_path = dataset.test.path
    
    train_df, test_df = load_timeseries_dataset(dataset)
    train_tsdf = TimeSeriesDataFrame(train_df)
    test_tsdf = TimeSeriesDataFrame(test_df)

    metadata = {
        "freq": dataset_config["freq"],
        "seasonality": dataset_config["seasonality"],
        "prediction_length": dataset_config["forecast_horizon_in_steps"],
    }

    print(f"Loaded {task_name}")
    return train_tsdf, test_tsdf, metadata


def visualize_predictions(
        df_pred_history: pd.DataFrame,
        task_name: str,
        item_ids: list[int],
        save_dir: Path,
        pipeline = None,
        filename_suffix: str | None = None,
    ):

    from tabpfn_ts.data.utils import plot_pred_and_actual_ts

    # Load the task dataset
    train_tsdf, test_tsdf, metadata = get_task_dataset(task_name)
    
    # Get corresponding item_ids
    dataset_all_item_ids = train_tsdf.item_ids
    repeated_id_to_item_id = {i: dataset_all_item_ids[i] for i in range(len(dataset_all_item_ids))}
    selected_item_ids = [repeated_id_to_item_id[i] for i in item_ids]

    selected_train_tsdf = train_tsdf.loc[selected_item_ids]
    selected_test_tsdf = test_tsdf.loc[selected_item_ids]

    # Visualize the predictions from history
    pred_from_history = df_pred_history[df_pred_history["repeated_item_id"].isin(item_ids)]
    pred_from_history = pred_from_history.set_index("repeated_item_id")[["predictions", "0.1", "0.9"]]
    adapted_pred_from_history = selected_test_tsdf.copy()
    for repeated_item_id in pred_from_history.index:
        predictions_by_item_id = pred_from_history.loc[repeated_item_id]
        item_id = repeated_id_to_item_id[repeated_item_id]
        adapted_pred_from_history.loc[item_id, 'target'] = predictions_by_item_id["predictions"].values
        adapted_pred_from_history.loc[item_id, '0.1'] = predictions_by_item_id["0.1"].values
        adapted_pred_from_history.loc[item_id, '0.9'] = predictions_by_item_id["0.9"].values

    plot_pred_and_actual_ts(
        pred=adapted_pred_from_history,
        train=selected_train_tsdf,
        test=selected_test_tsdf,
        show_points=False,
        save_path=save_dir / f"predictions_from_history{filename_suffix}.png"
    )

    # Visualize the predictions from inference (if provided, may take a while)
    if pipeline:
        pred_train = True
        pred = pipeline.predict(
            train_tsdf=selected_train_tsdf,
            predict_length=metadata["prediction_length"],
            quantile_config=[0.1, 0.9],
            predict_train=pred_train,
            dataset_metadata=metadata,
        )

        gt_tsdf = selected_test_tsdf if not pred_train \
            else pd.concat([selected_train_tsdf, selected_test_tsdf])
        
        plot_pred_and_actual_ts(
            pred=pred,
            train=selected_train_tsdf,
            test=gt_tsdf,
            show_points=False,
            save_path=save_dir / f"predictions_from_inference{filename_suffix}.png"
        )


def main():
    parser = ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('-p', "--pipeline", type=str, required=True)
    parser.add_argument('-d', '--history-dir', type=str, required=True)
    parser.add_argument('--rerun-inference', action='store_true')
    parser.add_argument('--filename-suffix', type=str, default=None)
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    history = extract_history(history_dir)
    task_name = history["task_name"]
    df_pred_history = history["predictions_history"]

    print(f"Task: {task_name}")

    # Compute the individual MASE and the distribution
    individual_mase = df_pred_history.groupby("repeated_item_id").apply(compute_individual_mase)
    desc_individual_mase = individual_mase.sort_values(ascending=False)
    save_mase_dist_plot(desc_individual_mase, save_path=history_dir / "mase_dist.png")

    worst_k_item_ids = desc_individual_mase.index[:args.k]
    print("worst_k_item_ids (repeated_item_id)", worst_k_item_ids)

    if args.rerun_inference:
        pipeline = get_pipeline(args.pipeline)

    visualize_predictions(
        df_pred_history=df_pred_history,
        task_name=task_name,
        item_ids=worst_k_item_ids,
        save_dir=history_dir,
        pipeline=pipeline if args.rerun_inference else None,
        filename_suffix=args.filename_suffix,
    )

    print(f"Done")

if __name__ == "__main__":
    main()
