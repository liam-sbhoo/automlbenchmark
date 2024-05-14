import logging
from importlib.metadata import version
import numpy as np
import pandas as pd

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import load_timeseries_dataset

from tabpfn import TabPFNRegressor

TABPFN_VERSION = version("tabpfn")


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

    model = TabPFNRegressor()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    # Crucial for the result to be interpreted as TimeSeriesResults
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id)[:len(test_y)],
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error)[:len(test_y)],
    )

    return result(
        output_file=config.output_predictions_file,
        predictions=pred,
        truth=test_y.values,
        target_is_encoded=False,
        models_count=1,
        training_duration=2,
        predict_duration=3,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)