#!/bin/bash -l
for test_name in test_1
do
  for actor_lr in 0.001 0.01 0.05 
  do
    for critic_lr in 0.001 0.01 0.05 0.1
    do
      ./gpu_batch.sh $test_name $actor_lr $critic_lr
    done
  done
done
