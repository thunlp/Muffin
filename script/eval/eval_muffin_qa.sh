###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===


base_dir=$1
to_process_ckpt_list="$1 "
# to_process_ckpt_list+=" $base_dir/checkpoint-40 $base_dir/checkpoint-80 $base_dir/checkpoint-120 $base_dir/checkpoint-160"
# to_process_ckpt_list+=" $base_dir/checkpoint-200 $base_dir/checkpoint-600 $base_dir/checkpoint-1000 $base_dir/checkpoint-1400 $base_dir/checkpoint-1800 $base_dir/checkpoint-2200 $base_dir/checkpoint-2600 $base_dir/checkpoint-3000"
# to_process_ckpt_list+=" $base_dir/checkpoint-400 $base_dir/checkpoint-800 $base_dir/checkpoint-1200 $base_dir/checkpoint-1600 $base_dir/checkpoint-2000 $base_dir/checkpoint-2400 $base_dir/checkpoint-2800 $base_dir/checkpoint-3200"
# to_process_ckpt_list+=" $base_dir/checkpoint-3600 $base_dir/checkpoint-4000 $base_dir/checkpoint-4400 $base_dir/checkpoint-4800 $base_dir/checkpoint-5200 $base_dir/checkpoint-5600 $base_dir/checkpoint-6000 $base_dir/checkpoint-6400"

# ===========> LLaVA Test Set <==============

answer_file_name="llava_test_answer.jsonl"

filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $ckpt/$answer_file_name
    if [[ ! -f $ckpt/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"


C=0
q_file=./playground/data/coco2014_val_qa_eval/qa90_questions_with_image.jsonl

for ckpt_path in $filered_to_process_ckpt_list;
do
    answer_file=$ckpt_path/$answer_file_name
    echo "PWD at `pwd` checkpoint: "$ckpt_path" output to: "$answer_file

    echo "Start generating answers for $ckpt_path"
    CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/muffin_vqa.py \
        --model-name $ckpt_path \
        --question-file $q_file \
        --answers-file  $answer_file &
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi
done
wait

# =========> unimm-bench <============

answer_file_name="unimm-bench_answer.jsonl"
eval_file_name="unimm-bench_gpt4_eval.jsonl"

filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $ckpt/$answer_file_name
    if [[ ! -f $ckpt/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"


C=0
q_file=/data/public/multimodal/multimodal_data/MMU_Benchmark/keep_400_vqa_eval.json

for ckpt_path in $filered_to_process_ckpt_list;
do
    answer_file=$ckpt_path/$answer_file_name
    echo "PWD at `pwd` checkpoint: "$ckpt_path" output to: "$answer_file

    CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/muffin_vqa.py \
        --model-name $ckpt_path \
        --question-file $q_file \
        --answers-file  $answer_file &
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi
done
wait


echo "========>Start GPT4 Evaluating<========"
bash ./script/eval/batch_gpt4_review.sh $base_dir 4
python ./eval/summarize_gpt_unimm-bench_review.py $base_dir > $base_dir/unimm-bench_scores.txt
python ./eval/summarize_gpt_llava_review.py $base_dir >> $base_dir/llava_test_scores.txt

# Print Log
echo Scores are:
cat $base_dir/unimm-bench_scores.txt
cat $base_dir/llava_test_scores.txt
echo done
