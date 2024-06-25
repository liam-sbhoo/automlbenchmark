import logging
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import load_timeseries_dataset
from autogluon.timeseries import TimeSeriesDataFrame
from frameworks.TabPFN.variants import NaiveTabPFNTimeSeriesPredictor


TABPFN_VERSION = version("tabpfn")
TABPFN_DEFAULT_QUANTILE = [i / 10 for i in range(1, 10)]


logger = logging.getLogger(__name__)


def split_time_series_to_X_y(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def run(dataset, config):
    logger.info(f"**** TabPFN TimeSeries [v{TABPFN_VERSION}] ****")

    # Initialize TimeSeriesPredictor
    tabpfn_model_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "show_progress": False,
    }
    predictor = NaiveTabPFNTimeSeriesPredictor(
        tabpfn_model_config=tabpfn_model_config,
        quantile_config=config.quantile_levels,
    )

    train_df, test_df = map(TimeSeriesDataFrame, load_timeseries_dataset(dataset))
    all_pred, meta = predictor.predict(train_df, test_df)
    
    # Crucial for the result to be interpreted as TimeSeriesResults
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id)[:test_df.shape[0]],
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error)[:test_df.shape[0]]
    )

    for q in config.quantile_levels:
        optional_columns[str(q)] = all_pred[str(q)]

    return result(
        output_file=config.output_predictions_file,
        predictions=all_pred["mean"],
        truth=test_df[dataset.target].values,
        target_is_encoded=False,
        models_count=1,
        training_duration=meta["training_duration"],
        predict_duration=meta["prediction_duration"],
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)