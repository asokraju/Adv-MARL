#!/bin/bash -l
for test_name in test_1
do
  for actor_lr in 0.01
  do
    for critic_lr in 0.01
    do 
      for random_seed in 10 20 30 40 50 60
      do
      ./gpu_batch.sh $test_name $actor_lr $critic_lr $random_seed
      done
    done
  done
done