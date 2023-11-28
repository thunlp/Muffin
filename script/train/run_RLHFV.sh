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

bsz=2
num_node=1
grad_acc=2

exp_name=$3
max_step=$8
save_step=$9

sft_data=$6
sft_data_weight=$7

dpo_use_average=${11}
dpo_token_weighted=${12}
dpo_token_weight=$4
dpo_beta=${10}
echo weight is $4 SFT weight is $SFT_weight beta is $dpo_beta

ref_model=$5

bash ./script/train/dpo_muffin_PATH.sh \
    $ref_model \
    $bsz \
    $grad_acc \
    not_used_param \
    $exp_name \
    beit3_large_patch16_448 \
    $sft_data \
    $sft_data_weight \
    $max_step \
    $save_step \
    $num_node \
    $root_dir \
    $rdzv_endpoint \
    $dpo_use_average \
    $dpo_token_weighted \
    $dpo_token_weight \
    $dpo_beta