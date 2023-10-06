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

if [[ $slave_or_master == "slave" ]]; then
    echo Slave rdzv_endpoint is "<${4}>"
    rdzv_endpoint=$4
fi


bsz=4
num_node=1
grad_acc=16

exp_name=$3
max_step=1600
save_step=400
sft_data="unimm-chat"

image_folder=$4

bash ./script/train/sft_muffin_PATH.sh \
    /home/yutianyu/Muffin_checkpoints/310m_pretrain_100k_SFT_M3IT_2800_then_M3IT-LVA-UniMM-SYNTHEDOG_2800/ \
    $bsz \
    $grad_acc \
    not_used_param \
    $exp_name \
    beit3_large_patch16_448 \
    $sft_data \
    100 \
    $max_step \
    $save_step \
    $num_node \
    $root_dir \
    $rdzv_endpoint \
    $image_folder