#!/bin/bash
# set -e

# Specify which Anaconda environment to use.
check_variable() {
    if [ -z "${ENV_FILE}" ]; then
        echo "Environment variable ENV_FILE is not set."
        return 1 # Return with a non-zero status to indicate failure
    fi
}
check_variable || return

export DATAROOT=$HOME/datasets/

ROOT='../..'
ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

conda env list | grep -q "^$ENV_NAME\s"
if [ $? -ne 0 ]; then
    echo "Creating conda environment $ENV_NAME"
    conda env create -f $ENV_FILE
fi
conda activate $ENV_NAME

VENV_DIR='.venv'
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment $VENV_DIR"
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install -r requirements.txt
    # cd $ROOT && pip install -e . && cd -
else
    source $VENV_DIR/bin/activate
fi
