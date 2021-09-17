#!/bin/bash -l
for test_name in adv_9_3_frequent
do
  for max_ep_len in 20
  do
    for n_ep_fixed in 1
    do
      for n_epochs in 1
      do
        for slow_lr in 0.002 0.005 0.01
        do
          for fast_lr in 0.01
          do
            for random_seed in 50 100
            do
              for consensus_freq in 1
              do
                ./crc_train.sh $test_name $max_ep_len $n_ep_fixed $n_epochs $slow_lr $fast_lr $random_seed $consensus_freq
              done
            done
          done
        done
      done
    done
  done
done
