# Preparation of Time-Series Benchmark

To benchmark against Shchur et al., “AutoGluon-TimeSeries.”, we need to prepare the data since it is not provided in the `openml/automlbenchmark`.

To do so:
```bash
 ./scripts/time-series-benchmark/setup_timeseries_benchmark.sh <dataset_dir
 ```

This will download the datasets and generate task configs for each of them.

- a collection of datasets are stored in `<dataset_dir>`
- `point_forecast.yaml` and `quantile_forecast.yaml` are generated and stored in `$HOME/.config/automlbenchmark/benchmarks`
