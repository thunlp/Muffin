
###===> Model config
sft_vision_tower=$6
llm_path=not_used
###<===

export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
sleep 5

image_folder=${14}

GPUS_PER_NODE=8
NUM_NODE=${11}
RDZV_ENDPOINT=${13}
RUNNER="torchrun --nnodes=${NUM_NODE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT}"

echo RUNNER=$RUNNER

###===> Start preparing data
sft_data=$7
sft_data_weight=$8
echo Data config: $sft_data $sft_data_weight
###<===


###===> Checkpointing
num_epoch=10 # not used indeed

num_save=10
save_step=${10}
max_step=$9
task_name=muffin_13b_SFT
exp_name=$5-$sft_data-$sft_data_weight-$sft_vision_tower
sft_output_dir=${12}/$task_name-$exp_name/checkpionts
sft_logging_dir=${12}/$task_name-$exp_name/log

echo "sft_output_dir="$sft_output_dir" sft_logging_dir="$sft_logging_dir
###<===

###===> SFT

pretrain_ckpt=$1
echo "Load from "$pretrain_ckpt

$RUNNER ./muffin/train/train_mem_muffin.py \
    --model_name_or_path $pretrain_ckpt \
    --image_folder $image_folder \
    --vision_tower $sft_vision_tower \
    --pretrain_mm_mlp_adapter not_used \
    --fully_tune True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir $sft_output_dir \
    --num_train_epochs $num_epoch \
    --per_device_train_batch_size $2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $save_step \
    --save_total_limit $num_save \
    --data_source_names  $sft_data \
    --data_source_weights $sft_data_weight \
    --max_steps $max_step \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir $sft_logging_dir \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $5 \
    --dataloader_num_workers 10
###<===
