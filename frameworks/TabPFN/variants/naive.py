import logging
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import torch

from frameworks.shared.utils import Timer
from tabpfn import TabPFNRegressor
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from autogluon.timeseries import TimeSeriesDataFrame

from dev.utils.dataset import TimeSeriesDatasetLoader


logger = logging.getLogger(__name__)
split_ts_dataset_to_X_y = TimeSeriesDatasetLoader.split_ts_dataset_to_X_y


class NaiveTabPFNTimeSeriesPredictor:

    DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, tabpfn_model_config: dict, quantile_config: list = DEFAULT_QUANTILE_CONFIG):
        self.model = TabPFNRegressor(**tabpfn_model_config)
        self.quantile_config = quantile_config

    def predict(self, train_df: TimeSeriesDataFrame, test_df: TimeSeriesDataFrame):
        all_pred = {str(q): [] for q in ["mean"] + self.quantile_config}

        training_duration = 0
        with Timer() as overall:
            for item_id in tqdm(train_df.item_ids):
                train_df_item = train_df.loc[item_id]
                test_df_item = test_df.loc[item_id]

                train_X, train_y = split_ts_dataset_to_X_y(train_df_item)
                test_X, _ = split_ts_dataset_to_X_y(test_df_item)

                if train_X.shape[1] == 0:
                    # Use a counter as the feature
                    train_X = np.arange(len(train_X)).reshape(-1, 1)
                    test_X = np.arange(len(test_X)).reshape(-1, 1)

                is_constant_value = (train_y.nunique() == 1)
                if not is_constant_value:
                    with Timer() as training:
                        self.model.fit(train_X, train_y)
                    training_duration += training.duration

                    pred = self.model.predict_full(test_X)

                else:
                    # If train_y is constant, we return the constant value from the training set
                    # For quantile prediction, we assume that the uncertainty follows a standard normal distribution
                    logger.info(f"Found time-series with constant target")

                    mean_constant = train_y.iloc[0]
                    pred = {"mean": np.full(len(test_X), mean_constant)}

                    quantile_pred_with_uncertainty = norm.ppf(self.quantile_config, loc=mean_constant, scale=1)
                    for i, q in enumerate(self.quantile_config):
                        pred[f"quantile_{q:.2f}"] = np.full(len(test_X), quantile_pred_with_uncertainty[i])

                    logger.info(f"Prediction: mean_constant={mean_constant:.2f}, quantile_pred={quantile_pred_with_uncertainty}")

                all_pred["mean"].append(pred["mean"])

                # Get quantile predictions
                for q in self.quantile_config:
                    if q in self.DEFAULT_QUANTILE_CONFIG:
                        quantile_pred = pred[f"quantile_{q:.2f}"]   # (n_horizon, )
                    else:
                        criterion: FullSupportBarDistribution = pred["criterion"]
                        logits = torch.tensor(pred["logits"])
                        quantile_pred = criterion.icdf(logits, q).numpy()   # (n_horizon, )

                    all_pred[str(q)].append(quantile_pred)

            # Concatenate all predictions
            for k in all_pred.keys():
                all_pred[k] = np.concatenate(all_pred[k], axis=0)   # (n_item * n_horizon,)

        prediction_duration = overall.duration - training_duration

        meta = {
            "training_duration": training_duration,
            "prediction_duration": prediction_duration,
        }

        return all_pred, meta

