#!/bin/bash
# set -e

HOSTNAME=$(hostname)
ROOT=$(git rev-parse --show-toplevel)

if [ $? -ne 0 ]; then
    echo "Could not find root directory. Exiting."
    return 1
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $HOSTNAME == "c3230" ]]; then
    export DATAROOT=$HOME/datasets/
    export ENV_FILE=$ROOT/env/py310_pyg230_cu115.yaml
elif [[ $HOSTNAME == gcn* ]]; then
    export DATAROOT=$HOME/datasets/
    export ENV_FILE=$ROOT/env/py310_pyg230_cu117.yaml
else
    echo "Unknown hostname $HOSTNAME". Exiting.
    return 1
fi

ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

conda env list | grep -q "^$ENV_NAME\s"
if [ $? -ne 0 ]; then
    echo "Environment $ENV_NAME not found. Exiting."
    return 1
else 
    conda activate $ENV_NAME
fi

VENV_DIR='.venv'
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment $VENV_DIR"
    python3 -m venv $VENV_DIR --system-site-packages
    source $VENV_DIR/bin/activate
    pip install -r env/requirements.txt
    cd $ROOT/lib/rail1/ && pip install -e . && cd -
else
    source $VENV_DIR/bin/activate
fi
