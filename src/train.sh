#!/bin/bash

assist_dataset_names="assist2009_pid assist2012_pid assist2017_pid"
diff_algebra_dataset_names="algebra2005_pid_diff algebra2006_pid_diff"
algebra_dataset_names="algebra2005_pid algebra2006_pid"

python \
train.py \
--model_fn monaconvbert4kt_plus_diff_rmse.pth \
--model_name monaconvbert4kt_plus_diff \
--dataset_name algebra2006_pid_diff \
--num_encoder 12 \
--batch_size 128 \
--grad_acc True \
--grad_acc_iter 4 \
--use_leakyrelu True \
--crit rmse \
--fivefold True \
--n_epochs 1000

for algebra_dataset_name in ${algebra_dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_rmse.pth \
    --model_name monaconvbert4kt_plus \
    --dataset_name ${algebra_dataset_name} \
    --num_encoder 12 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --use_leakyrelu True \
    --crit rmse \
    --fivefold True \
    --n_epochs 1000
done

