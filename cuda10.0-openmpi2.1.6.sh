#!/bin/sh
#$ -l rt_F=2
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

ENV_NAME=`basename $JOB_NAME .sh`-env

source /etc/profile.d/modules.sh

module load python/3.6/3.6.5

module load cuda/10.0/10.0.130.1
module load cudnn/7.6/7.6.4
module load nccl/2.4/2.4.8-1

# cuda-aware openmpi/2.1.6
module load openmpi/2.1.6

# GCC 7.3.0
# module load gcc/7.3.0
export PATH=/apps/gcc/7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/gcc/7.3.0/lib64:$LD_LIBRARY_PATH

#-------------------------------------------------------------
# Setup python3-venv
#-------------------------------------------------------------
if [ ! -e $ENV_NAME ]; then
    python3 -m venv $ENV_NAME
fi

source $ENV_NAME/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools

# compatibility: gcc/7.3.0 cuda/10.0
pip3 install "tensorflow-gpu>=1.13,<1.14" keras torch torchvision
HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_HOME=$NCCL_HOME \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1 \
    pip3 install --no-cache-dir horovod

# Confirm
pip3 list
module list
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
    "pytorch_mnist" | "tensorflow_mnist" | "tensorflow_word2vec" | "keras_mnist" )
	mpirun ${MPIOPTS} python3 ${HOROVOD_EXAMPLES}/$1.py
	;;
    *)
	echo "Usage: $0 pytorch_mnist | tensorflow_mnist | tensorflow_word2vec | keras_mnist"
	;;
esac

endtime=`date "+%Y/%m/%d %H:%M"`

echo "Start Time : $starttime"
echo "End   Time : $endtime"

deactivate
