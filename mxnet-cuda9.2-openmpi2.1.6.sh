#!/bin/sh
#$ -l rt_F=2
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

ENV_NAME=`basename $JOB_NAME .sh`-env

source /etc/profile.d/modules.sh

module load python/3.6/3.6.5

module load cuda/9.2/9.2.148.1
module load cudnn/7.6/7.6.4
module load nccl/2.4/2.4.8-1

# cuda-aware openmpi/2.1.6
module load openmpi/2.1.6

#-------------------------------------------------------------
# Setup python3-venv
#-------------------------------------------------------------
if [ ! -e $ENV_NAME ]; then
    python3 -m venv $ENV_NAME
fi

source $ENV_NAME/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools

pip3 install mxnet-cu92
HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_WITH_MXNET=1 \
    pip3 install --no-cache-dir horovod

# Confirm
pip3 list
module list -l
nvidia-smi

#-------------------------------------------------------------
# Run Sample
#-------------------------------------------------------------
HOROVOD_EXAMPLES=horovod/examples

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node"

starttime=`date "+%Y/%m/%d %H:%M"`

case "$1" in
    "mxnet_mnist" )
	mpirun ${MPIOPTS} python3 ${HOROVOD_EXAMPLES}/$1.py
	;;
    *)
	echo "Usage: $0 mxnet_mnist"
	;;
esac

endtime=`date "+%Y/%m/%d %H:%M"`

echo "Start Time : $starttime"
echo "End   Time : $endtime"

deactivate
