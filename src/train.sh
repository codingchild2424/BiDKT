#!/bin/bash

python \
train.py \
--model_fn bert4kt_plus.pth \
--model_name bert4kt_plus \
--dataset_name assist2009_pid \
--num_encoder 12 \
--batch_size 256 \
--grad_acc True \
--grad_acc_iter 2 \
--use_leakyrelu True \
--fivefold True \
--n_epochs 1000

python \
train.py \
--model_fn monabert4kt_plus.pth \
--model_name monabert4kt_plus \
--dataset_name assist2009_pid \
--num_encoder 12 \
--batch_size 256 \
--grad_acc True \
--grad_acc_iter 2 \
--use_leakyrelu True \
--fivefold True \
--n_epochs 1000

python \
train.py \
--model_fn convbert4kt_plus.pth \
--model_name convbert4kt_plus \
--dataset_name assist2009_pid \
--num_encoder 12 \
--batch_size 256 \
--grad_acc True \
--grad_acc_iter 2 \
--use_leakyrelu True \
--fivefold True \
--n_epochs 1000


