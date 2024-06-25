import logging
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import norm

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import load_timeseries_dataset, Timer

from tabpfn import TabPFNRegressor
from tabpfn.model.bar_distribution import FullSupportBarDistribution

TABPFN_VERSION = version("tabpfn")
TABPFN_DEFAULT_QUANTILE = [i / 10 for i in range(1, 10)]


logger = logging.getLogger(__name__)


def split_time_series_to_X_y(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def run(dataset, config):
    logger.info(f"**** TabPFN TimeSeries [v{TABPFN_VERSION}] ****")

    # Initialize TabPFN model
    model = TabPFNRegressor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        show_progress=False,
    )
    logger.info(f"Using device: {model.device}")

    train_df, test_df = load_timeseries_dataset(dataset)

    # Sort and group by item_id
    group_key = "item_id"
    train_df.sort_values(by=[group_key], inplace=True)
    test_df.sort_values(by=[group_key], inplace=True)
    train_grouped = train_df.groupby(group_key)
    test_grouped = test_df.groupby(group_key)
    assert len(train_grouped) == len(test_grouped)

    # Perform prediction for each time-series
    all_pred = {str(q): [] for q in ["mean"] + config.quantile_levels}
    with Timer() as predict:
        count = 5
        for (train_id, train_group), (test_id, test_group) in tqdm(zip(train_grouped, test_grouped),
                                                               total=len(train_grouped),
                                                               desc="Processing Groups"):
            assert train_id == test_id

            train_group.drop(columns=[group_key], inplace=True)
            test_group.drop(columns=[group_key], inplace=True)

            train_X, train_y = split_time_series_to_X_y(train_group)
            test_X, _ = split_time_series_to_X_y(test_group)

            is_constant_value = (train_y.nunique() == 1)

            # TabPFN fit and predict at the same time (single forward pass)
            if not is_constant_value:
                with Timer() as training:
                    model.fit(train_X, train_y)
                pred = model.predict_full(test_X)

            else:
                # If train_y is constant, we return the constant value from the training set
                # For quantile prediction, we assume that the uncertainty follows a standard normal distribution
                mean_constant = train_y.iloc[0]
                pred = {"mean": np.full(len(test_X), mean_constant)}

                quantile_pred_with_uncertainty = norm.ppf(config.quantile_levels, loc=mean_constant, scale=1)
                for i, q in enumerate(config.quantile_levels):
                    pred[f"quantile_{q:.2f}"] = np.full(len(test_X), quantile_pred_with_uncertainty[i])

                logger.info(f"Found time-series with constant target")
                logger.info(f"Prediction: mean_constant={mean_constant:.2f}, quantile_pred={quantile_pred_with_uncertainty}")

            all_pred["mean"].append(pred["mean"])

            # Get quantile predictions
            for q in config.quantile_levels:
                if q in TABPFN_DEFAULT_QUANTILE:
                    quantile_pred = pred[f"quantile_{q:.2f}"]   # (n_horizon, )
                else:
                    criterion: FullSupportBarDistribution = pred["criterion"]
                    logits = torch.tensor(pred["logits"])
                    quantile_pred = criterion.icdf(logits, q).numpy()   # (n_horizon, )

                all_pred[str(q)].append(quantile_pred)

        # Concatenate all quantile predictions
        for k in all_pred.keys():
            all_pred[k] = np.concatenate(all_pred[k], axis=0)   # (n_item * n_horizon,)

    training_duration = training.duration if not is_constant_value else 0.0
    predict_duration = predict.duration - training_duration

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
        training_duration=training_duration,
        predict_duration=predict_duration,
        optional_columns=pd.DataFrame(optional_columns),
    )


if __name__ == '__main__':
    call_run(run)