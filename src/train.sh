#!/bin/bash

python \
train.py \
--model_fn monacobert4kt_rasch.pth \
--model_name monaconvbert4kt_rasch \
--dataset_name assist2009_pid \
--num_encoder 12 \
--batch_size 256 \
--grad_acc True \
--grad_acc_iter 2 \
--use_leakyrelu True \
--fivefold True \
--n_epochs 1000

