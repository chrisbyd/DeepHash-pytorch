#!/usr/bin/env bash
dataset=cifar10   # "dataset": "nuswide81" "cifar10" "imagenet",
 #
 # LCDSH batch_size=800
 #

model_name=DSDH # HashNet.py  PCDH.py LCDSH GreedyHash
bit_list=[16,32,48,64]
num_epochs=200
batch_size=800
gpu_deivice=0
test_batch_size=1000
test_map=10
machine_name=1080
echo "Start training  "

python train.py  $model_name $num_epochs $bit_list $batch_size $dataset $gpu_deivice $machine_name $test_map $test_batch_size

