import logging
from importlib.metadata import version
import numpy as np
import pandas as pd
from pathlib import Path

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer, load_timeseries_dataset

from tabpfn_ts.data import TimeSeriesDataFrame
from tabpfn_ts.pipeline import PredictionPipeline


logger = logging.getLogger(__name__)

# Temporary (for development)
TABPFN_TS_CONFIG_FILE_DIR = Path("/home/hoos/hoos-time/playground/tabpfn-time-series/tabpfn_ts/pipeline/config")


def run(dataset, config):
    logger.info(f"\n**** TabPFN-TS [v{version('tabpfn_ts')}] ****\n")
    
    train_df, test_df = load_timeseries_dataset(dataset)
    train_data = TimeSeriesDataFrame(
        train_df,
        id_column=dataset.id_column,
        timestamp_column=dataset.timestamp_column,
    )

    framework_params=config.framework_params
    tabpfn_ts_config_file = TABPFN_TS_CONFIG_FILE_DIR / framework_params["config_file"]
    pipeline = PredictionPipeline.from_config_file(tabpfn_ts_config_file)

    with Timer() as predict:
        pred = pipeline.predict(
            train_tsdf=train_data,
            predict_length=dataset.forecast_horizon_in_steps,
            quantile_config=config.quantile_levels,
        )

    # Add columns necessary for the metric computation + quantile forecast to `optional_columns`
    optional_columns = dict(
        repeated_item_id=np.load(dataset.repeated_item_id),
        repeated_abs_seasonal_error=np.load(dataset.repeated_abs_seasonal_error),
    )
    for q in config.quantile_levels:
        optional_columns[str(q)] = pred[q].values

    return result(
        output_file=config.output_predictions_file,
        predictions=pred["target"],
        truth=test_df[dataset.target].values,
        target_is_encoded=False,
        models_count=1,
        training_duration=0,
        predict_duration=predict.duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)
