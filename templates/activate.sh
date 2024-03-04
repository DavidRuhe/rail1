#!/bin/bash
# set -e

# Structure:
# bin/
# env/environment_name.yaml
# src/project_name/activate.sh

# Specify which Anaconda environment to use.
ENV_FILE='c3230.yaml'
ROOT='../..'
ENV_NAME=$(grep 'name:' $ROOT/env/$ENV_FILE | cut -d ' ' -f 2)

conda env list | grep -q "^$ENV_NAME\s"
if [ $? -ne 0 ]; then
    echo "Creating conda environment $ENV_NAME"
    conda env create -f $ROOT/env/$ENV_FILE
fi
conda activate $ENV_NAME

VENV_DIR='.venv'
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment $VENV_DIR"
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install -r requirements.txt
    cd $ROOT && pip install -e . && cd -
else
    source $VENV_DIR/bin/activate
fi
