import logging
from pathlib import Path
import yaml
import hashlib
import json
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from pprint import pprint

from amlb.datasets.file import TimeSeriesDataset
from amlb.utils.core import Namespace
from autogluon.timeseries import TimeSeriesDataFrame
from frameworks.shared.utils import load_timeseries_dataset


logger = logging.getLogger(__name__)


class TimeSeriesDatasetLoader:

    TIMESTAMP_FEATURE = "timestamp"
    TMP_CACHE_DIR = Path("./tmp_cache")
    DATASETS_METADATA_CACHE_FILE = TMP_CACHE_DIR / "datasets_metadata_cache.json"

    def __init__(self, config_file: Path):
        self._datasets_config = self._load_dataset_config(config_file)
        
        if self.DATASETS_METADATA_CACHE_FILE.exists():
            self._datasets_metadata_db: dict[str, dict] = json.loads(self.DATASETS_METADATA_CACHE_FILE.read_text())
        else:
            self.DATASETS_METADATA_CACHE_FILE.touch()
            self._datasets_metadata_db: dict[str, dict] = {}

        config_hash = self._generate_datasets_config_hash(self._datasets_config)
        if config_hash in self._datasets_metadata_db:
            self._datasets_metadata = self._datasets_metadata_db[config_hash]
        else:
            self._datasets_metadata = self._generate_datasets_metadata()
            self._datasets_metadata_db[config_hash] = self._datasets_metadata
            self.DATASETS_METADATA_CACHE_FILE.write_text(json.dumps(self._datasets_metadata_db, indent=4))
    
    @property
    def datasets_available(self) -> pd.DataFrame:
        available_datasets = pd.DataFrame.from_dict(self._datasets_metadata, orient='index').sort_index()
        return available_datasets

    def load_dataset(
            self,
            name: str,
            num_fold: int = 0,
            num_series: int | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if name not in self._datasets_config:
            raise ValueError(f"Dataset {name} not available")

        logger.info(f"Loading dataset {name}")

        d_config = self._datasets_config[name]['dataset']

        d = TimeSeriesDataset(
            path=d_config["path"],
            fold=num_fold,
            target=d_config["target"],
            features=self.TIMESTAMP_FEATURE,
            cache_dir=self.TMP_CACHE_DIR.resolve(),
            config=Namespace(d_config | {"name": name}),
        )
        d.train_path = d.train.path
        d.test_path = d.test.path

        def preprocess_ts_df(df: pd.DataFrame) -> TimeSeriesDataFrame:
            if num_series is not None:
                df = df.iloc[:num_series]
            return update_ts_freq(TimeSeriesDataFrame(df))

        train_df, test_df = map(preprocess_ts_df, load_timeseries_dataset(d))
        return train_df, test_df

    @staticmethod
    def split_ts_dataset_to_X_y(df, target_col="target"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    @staticmethod
    def visualize_target(df, target_col="target"):
        # Visualize target over timestamp
        plt.figure(figsize=(10, 6))
        plt.plot(df[TimeSeriesDatasetLoader.TIMESTAMP_FEATURE], df[target_col], label=target_col)
        plt.xlabel('Timestamp')
        plt.ylabel(target_col)
        plt.title(f'{target_col} over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def _load_dataset_config(config_file: Path) -> dict[str, dict]:
        config_in_list = yaml.safe_load(config_file.read_text())
        return {c["name"]: c for c in config_in_list}
    
    @staticmethod
    def _generate_datasets_config_hash(datasets_config: dict[str, dict]):
        config_str = str(datasets_config)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _generate_datasets_metadata(self):
        # Go through each dataset and load it, then extract metadata
        logger.info("Generating datasets metadata")
        metadata = {}
        
        def process_dataset(name, config):
            train_df, _ = self.load_dataset(name, num_fold=0)
            current_metadata = {
                "num_series": len(train_df["item_id"].unique()),
                "num_timesteps": len(train_df),
                "seasonality": config['dataset']['seasonality'],
                "forecast_horizon": config['dataset']['forecast_horizon_in_steps'],
                "frequency": config['dataset']['freq'],
            }
            return name, current_metadata
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_dataset, name, config) for name, config in self._datasets_config.items()]
            for future in concurrent.futures.as_completed(futures):
                name, future_metadata = future.result()
                metadata[name] = future_metadata
        
        return metadata

def update_ts_freq(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    tsdf.index.levels[1].freq = pd.infer_freq(tsdf.index.levels[1])
    return tsdf
