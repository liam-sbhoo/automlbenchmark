#!/bin/bash

# This script could be used to download and prepare the datasets for time-series forecasting.

DATASETS_DIR=$1

# Assert that the datasets directory is provided
if [ -z "$DATASETS_DIR" ]; then
    echo "Please provide the datasets directory as an argument."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DOWNLOAD_DATASETS_SCRIPT=$SCRIPT_DIR/download_datasets.py
ROOT_DIR=$SCRIPT_DIR/../..

# Download M3C dataset if it does not exist
mkdir -p $DATASETS_DIR
if [ ! -f $DATASETS_DIR/M3C.xls ]; then
    wget https://forecasters.org/data/m3comp/M3C.xls -P $DATASETS_DIR
fi

# Create a temporary virtual environment to run the script
VENV_NAME=temp_venv
if [ -d "$VENV_NAME" ]; then
    echo "Cleaning up existing virtual environment..."
    rm -rf $VENV_NAME
fi
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install gluonts pandas orjson pyyaml xlrd awscli joblib
python $DOWNLOAD_DATASETS_SCRIPT -d $DATASETS_DIR
deactivate
rm -rf $VENV_NAME

# Verify the datasets
python $ROOT_DIR/runbenchmark.py SeasonalNaive timeseries