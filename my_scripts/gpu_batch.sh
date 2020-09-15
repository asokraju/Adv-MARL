#!/bin/bash

TEST_NAME=$1
A_LR=$2
C_LR=$3

#Creating necessary folders to save the results of the experiment
PARENT_DIR="$(dirname $PWD)"             #file is inside this  directory
EXEC_DIR=$PWD                            #gpu_batch script is inside this dir
TEST_NAME_DIR="test_name=${TEST_NAME}"   #directory with test name
A_LR_DIR="a_lr=${A_LR}"               #directory for parameter gamma name
C_LR_DIR="c_lr=${C_LR}"                #directory for parameter time steps name

mkdir -p $TEST_NAME_DIR                  #making a directory with test name
RESULTS_DIR=${EXEC_DIR}/${TEST_NAME_DIR} #Directory for results
cd $RESULTS_DIR                          #we are inside the results_dir

mkdir -p $A_LR_DIR                      #making a directory for parameter gamma name
cd $A_LR_DIR
mkdir -p $C_LR_DIR                         #making a directory for parameter time steps name
cd $C_LR_DIR

export run_exec=$PARENT_DIR/learner.py #python script that we want to run
#export run_exec=/afs/crc.nd.edu/user/k/kkosaraj/kristools/microgrid_dcbf.py
export run_flags="--actor_lr=${A_LR} --critic_lr=${C_LR} --summary_dir='$PWD' > out.txt"  #flags for the script

echo "#!/bin/bash" > job.sh
echo "#$ -M kkosaraj@nd.edu" >> job.sh  # Email address for job notification
echo "#$ -m abe"   >> job.sh         # Send mail when job begins, ends and aborts
echo "#$ -q gpu" >> job.sh                                      # which queue to use: debug, long, gpu
echo "#$ -l gpu_card=1" >>job.sh                                # need if we use gpu queue
#echo "#$ -pe smp 1" >> job.sh
echo "#$ -N DDPG_actor_lr=${A_LR}_critic_lr=${C_LR}" >> job.sh             # name for job
echo "#$ -o info" >> job.sh
echo "module load conda" >> job.sh                              #loading the desired modules
echo "module load cuda" >> job.sh
echo "module load cudnn" >> job.sh
echo "conda activate tf_gpu_krishna" >> job.sh

echo "python $run_exec $run_flags" >> job.sh

qsub job.sh


#DO YOU SEE ME