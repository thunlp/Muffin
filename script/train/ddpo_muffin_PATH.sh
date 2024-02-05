
###===> Model config
sft_vision_tower=$6
llm_path=not_used
###<===

export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
sleep 5


GPUS_PER_NODE=8
NUM_NODE=${11}
RDZV_ENDPOINT=${13}
RUNNER="torchrun --nnodes=${NUM_NODE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT}"

echo RUNNER=$RUNNER

###===> Start preparing data
data_dir=$7
ref_name=$8
echo Data config: $data_dir $ref_name
###<===


###===> Checkpointing
num_epoch=10 # not used indeed

num_save=16
save_step=${10}
max_step=$9
task_name=muffin_13b_DPO
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
    --image_folder not_used \
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
    --data_source_names  '' \
    --data_source_weights '' \
    --data_dir $data_dir \
    --ref_name $ref_name \
    --max_steps $max_step \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --logging_dir $sft_logging_dir \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $5 \
    --dataloader_num_workers 10 \
    --dpo_use_average ${14} \
    --dpo_token_weighted ${15} \
    --dpo_token_weight ${16} \
    --dpo_beta ${17}
###<===

