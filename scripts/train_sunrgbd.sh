#/bin/bash
nohup python3 -u train.py --gpus 1 --cfg_file configs/frustum_sunrgbd.yaml --max_epochs 120 --dataset_name sunrgbd 2>&1 >sun.log &