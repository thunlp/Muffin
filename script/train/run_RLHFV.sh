#!/bin/bash

#SBATCH --partition=gpu3-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=g3013
#SBATCH --output=./_temp/slurm_output/%j.%x.out

echo Working Directory at `pwd`
echo Bash at `which bash`
echo Python at `which python`

export PYTHONPATH=$PYTHONPATH:`realpath .`

nvidia-smi

root_dir=$1 # directory to save log and checkpoints
slave_or_master=$2


MASTER_ADDR=`hostname`
MASTER_PORT=13245
rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT

echo $slave_or_master local_addr=$rdzv_endpoint

bsz=1
num_node=1
grad_acc=1

exp_name=$3
max_step=$8
save_step=$9

data_dir=$6
ref_name=$7

dpo_use_average=${11}
dpo_token_weighted=${12}
dpo_token_weight=$4
dpo_beta=${10}
echo ddpo weight is $4 beta is $dpo_beta

ref_model=$5

bash ./script/train/ddpo_muffin_PATH.sh \
    $ref_model \
    $bsz \
    $grad_acc \
    not_used_param \
    $exp_name \
    beit3_large_patch16_448 \
    $data_dir \
    $ref_name \
    $max_step \
    $save_step \
    $num_node \
    $root_dir \
    $rdzv_endpoint \
    $dpo_use_average \
    $dpo_token_weighted \
    $dpo_token_weight \
    $dpo_beta
