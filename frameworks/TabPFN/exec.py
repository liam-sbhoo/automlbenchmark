import logging
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import load_timeseries_dataset, Timer

from tabpfn import TabPFNRegressor
from tabpfn.model.bar_distribution import FullSupportBarDistribution

TABPFN_VERSION = version("tabpfn")
TABPFN_DEFAULT_QUANTILE = [i / 10 for i in range(1, 10)]


logger = logging.getLogger(__name__)


def split_time_series_dataset_to_X_y(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def run(dataset, config):
    logger.info(f"**** TabPFN TimeSeries [v{TABPFN_VERSION}] ****")

    train_df, test_df = load_timeseries_dataset(dataset)

    # (Temporary) Loop through all items in the dataset
    selected_item_id = train_df[dataset.id_column].unique()[0]
    train_df_single = train_df[train_df[dataset.id_column] == selected_item_id].drop(
        columns=[dataset.id_column])
    test_df_single = test_df[test_df[dataset.id_column] == selected_item_id].drop(
        columns=[dataset.id_column])

    train_X, train_y = split_time_series_dataset_to_X_y(
        train_df_single, target_col=dataset.target
    )
    test_X, test_y = split_time_series_dataset_to_X_y(
        test_df_single, target_col=dataset.target
    )

    # TODO: Call groupby to split the dataset into multiple time-series (items)
    # TODO: Implement parallization for TabPFN (imagine config.cores > 1)

    model = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {model.device}")

    with Timer() as predict:
        # TabPFN fit and predict at the same time (single forward pass)
        model.fit(train_X, train_y)
        pred = model.predict_full(test_X)

    # Crucial for the result to be interpreted as TimeSeriesResults
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id)[:len(test_y)],
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error)[
                                    :len(test_y)],
    )

    # Get quantile predictions
    for q in config.quantile_levels:
        if q in TABPFN_DEFAULT_QUANTILE:
            quantile_pred = pred[f"quantile_{q:.2f}"]

        else:
            criterion: FullSupportBarDistribution = pred["criterion"]
            logits = torch.tensor(pred["logits"])
            quantile_pred = criterion.icdf(logits, q).numpy()

        optional_columns[str(q)] = quantile_pred

    return result(
        output_file=config.output_predictions_file,
        predictions=pred["mean"],
        truth=test_y.values,
        target_is_encoded=False,
        models_count=1,
        training_duration=0.0,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)