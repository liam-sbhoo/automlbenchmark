#!/usr/bin/env bash
HERE=$(dirname "$0")
. "${HERE}/../shared/setup.sh" "${HERE}" true

# Temporary (for development)
TABPFN_TS_REPO_DIR="${HERE}/../../../tabpfn-time-series"
PIP install -e ${TABPFN_TS_REPO_DIR}

PY -c "from importlib.metadata import version; print(version('tabpfn-ts'))" >> "${HERE}/.setup/installed"
