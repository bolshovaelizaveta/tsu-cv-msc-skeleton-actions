#!/bin/bash
set -e

echo "======================================"
echo "Setting up STGCN++ (MMAction2)"
echo "======================================"

# Правильная структура как в репозитории
mkdir -p configs/skeleton/stgcnpp
mkdir -p configs/_base_
mkdir -p models

echo "Downloading STGCN++ config..."

wget -O configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py

echo "Downloading base runtime config..."

wget -O configs/_base_/default_runtime.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/default_runtime.py

echo "Downloading STGCN++ weights..."

wget -O models/stgcnpp_ntu60_xsub.pth \
https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth

echo "Checking structure..."

ls -R configs
ls -lh models

echo "Done ✔"