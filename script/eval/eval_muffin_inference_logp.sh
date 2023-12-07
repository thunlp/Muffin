###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===


ckpt_path=$1
base_dir=$2
to_process_tsv_list="$3 "

echo $to_process_tsv_list

# save_logp_name is the suffix to add to the logp file, we defaultly use 'dpo_with_rlhf-v-sft_logp_train'
save_logp_name='dpo_with_rlhf-v-sft_logp_train'

C=0

for tsv_file in $to_process_tsv_list;
do
    echo "PWD at `pwd` checkpoint: "$ckpt_path

    CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/muffin_inference_logp.py \
        --model-name $ckpt_path \
        --data-dir $base_dir \
        --tsv-file  $tsv_file \
        --logp-file $save_logp_name
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi
done