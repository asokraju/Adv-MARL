#!/bin/bash

TEST_NAME=$1
EP_LEN=$2
N_EP_FIXED=$3
N_EPOCHS=$4
SLOW_LR=$5
FAST_LR=$6
R_SEED=$7
CON_FREQ=$8

#Creating necessary folders to save the results of the experiment
PARENT_DIR="$(dirname $PWD)"             #file is inside this directory
EXEC_DIR=$PWD                            #gpu_batch script is inside this dir
TEST_NAME_DIR="${TEST_NAME}"             #directory with test name
EP_LEN_DIR="EpLen=${EP_LEN},epFixed=${N_EP_FIXED},epochs=${N_EPOCHS}"    #directory with episode length
LR_DIR="lr=${SLOW_LR},${FAST_LR}"        #directory with learning rates
R_SEED_DIR="seed=${R_SEED}"              #directory with random seeds

mkdir -p $TEST_NAME_DIR                  #making a directory with test name
cd $TEST_NAME_DIR                        #we are inside the test directory
mkdir -p $EP_LEN_DIR                     #making a directory for episode length
cd $EP_LEN_DIR                           #we are inside the episode length directory
mkdir -p $LR_DIR                         #making a directory for learning rate directory
cd $LR_DIR                               #we are inside the learning rate directory
mkdir -p $R_SEED_DIR                     #making a directory for random seeds
cd $R_SEED_DIR                           #we are inside the random seed directory

export run_exec=$PARENT_DIR/main.py      #python script that we want to run
export run_flags="--max_ep_len=${EP_LEN} --slow_lr=${SLOW_LR} --fast_lr=${FAST_LR} --summary_dir='$PWD' --random_seed=${R_SEED} --n_ep_fixed=${N_EP_FIXED} --consensus_freq=${CON_FREQ} --n_epochs=${N_EPOCHS} > out.txt"  #flags for the script

echo "#!/bin/bash" > job.sh
echo "#$ -M mfigura@nd.edu" >> job.sh  # Email address for job notification
echo "#$ -m abe"   >> job.sh         # Send mail when job begins, ends and aborts
#echo "#$ -q gpu" >> job.sh                                      # which queue to use: debug, long, gpu
#echo "#$ -l gpu_card=1" >>job.sh                                # need if we use gpu queue
echo "#$ -pe smp 1" >> job.sh
echo "#$ -N Job=${EP_LEN},${NN_LAYERS},${NN_UNITS},${A_LR},${C_LR},${TR_LR},${R_SEED}" >> job.sh             # name for job
echo "#$ -o info" >> job.sh
echo "module load conda" >> job.sh                              #loading the desired modules
echo "conda activate MARL_env" >> job.sh

echo "/afs/crc.nd.edu/user/m/mfigura/.conda/envs/MARL_env/bin/python $run_exec $run_flags" >> job.sh

qsub job.sh


#DO YOU SEE ME
