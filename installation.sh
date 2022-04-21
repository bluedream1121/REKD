#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

echo ""
echo ""

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.8 ******************"
conda create -y --name $conda_env_name python=3.8

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo "****************** Installing pytorch with cuda11.1 ******************"
conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge


# echo "****************** Installing pytorch with cuda10.2 ******************"
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch


echo ""
echo ""
echo "****************** Installing kornia, e2cnn, pandas, opencv, scipy, tqdm, matplotlib, jupyter ******************"
pip install tqdm kornia==0.5.2  scipy e2cnn pandas scikit-image opencv-python matplotlib jupyter


