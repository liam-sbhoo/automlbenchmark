#!/bin/bash

# This script could be used to prepare the datasets and tasks config for time-series forecast benchmark.

DATASETS_DIR=$1

# Assert that the datasets directory is provided
if [ -z "$DATASETS_DIR" ]; then
    echo "Please provide the datasets directory as an argument."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DOWNLOAD_DATASETS_SCRIPT=$SCRIPT_DIR/download_datasets.py
GENERATE_TASK_CONFIGS_SCRIPT=$SCRIPT_DIR/generate_task_configs.py
GENERATE_ID_2_TASK_MAPPING_SCRIPT=$SCRIPT_DIR/generate_array_id_to_task_mapping.py
ROOT_DIR=$SCRIPT_DIR/../..
BENCHMARK_FILES_OUTPUT_DIR=$HOME/.config/automlbenchmark/benchmarks

## Download M3C dataset if it does not exist
#mkdir -p $DATASETS_DIR
#if [ ! -f $DATASETS_DIR/M3C.xls ]; then
#    wget https://forecasters.org/data/m3comp/M3C.xls -P $DATASETS_DIR
#fi
wget https://forecasters.org/data/m3comp/M3C.xls -P ~/.gluonts/datasets

### Start of python venv
VENV_NAME=temp_venv
if [ -d "$VENV_NAME" ]; then
    echo "Cleaning up existing virtual environment..."
    rm -rf $VENV_NAME
fi
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install gluonts==0.14.4 pandas==1.5.3 orjson pyyaml xlrd awscli joblib

# Download datasets
python $DOWNLOAD_DATASETS_SCRIPT -d $DATASETS_DIR

# Generate tasks config
NUM_SERIES_THRESHOLD=10000
python $GENERATE_TASK_CONFIGS_SCRIPT -d $DATASETS_DIR -b $BENCHMARK_FILES_OUTPUT_DIR -s $NUM_SERIES_THRESHOLD
python $GENERATE_ID_2_TASK_MAPPING_SCRIPT $BENCHMARK_FILES_OUTPUT_DIR/point_forecast_skip_${NUM_SERIES_THRESHOLD}.yaml $BENCHMARK_FILES_OUTPUT_DIR

# Copy the generated task config to the benchmark directory
cp $HOME/.config/automlbenchmark/benchmarks/* $ROOT_DIR/resources/benchmarks

### End of python venv
deactivate
rm -rf $VENV_NAME