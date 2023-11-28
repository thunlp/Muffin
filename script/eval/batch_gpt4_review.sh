SOURCE_DIR=$1
limit=100
C=0
process_limit=2
force=no_force

while IFS= read -r -d '' -u 9
do
    # echo $REPLY
    if [[ $REPLY == *$prefix*unimm-bench_answer.jsonl ]]; then
        echo "EVAL unimm-bench "$REPLY
        if [[ $force == force ]]; then
            rm -f $REPLY.unimm-bench_gpt4_eval.jsonl
        fi
        python ./eval/eval_gpt_review_unimm-bench.py \
            --question ./data/unimm-bench.json \
            --answer $REPLY \
            --rule ./eval/data/rule.jsonfile \
            --output $REPLY.unimm-bench_gpt4_eval.jsonl \
            --limit $limit &
        sleep 5

        C=$((C+1))
        echo "C=$C"
        if [[ $C == $process_limit ]]; then
            echo "Wait for next iteration"
            C=0
            wait
        fi
    fi
done 9< <( find $SOURCE_DIR -type f -name "*qa*" -exec printf '%s\0' {} + )

wait

while IFS= read -r -d '' -u 9
do
    # echo $REPLY
    if [[ $REPLY == *$prefix*llava_test_answer.jsonl ]]; then
        if [[ $force == force ]]; then
            rm -f $REPLY.llava_test_gpt4.jsonl
        fi
        echo "EVAL qa90 "$REPLY
        python ./eval/eval_gpt_review_visual.py \
            --question ./eval/data/qa90_questions.jsonl \
            --context ./eval/data/caps_boxes_coco2014_val_80.jsonl \
            --answer-list \
            ./eval/data/qa90_gpt4_answer.jsonl \
            $REPLY \
            --rule ./eval/data/rule.jsonfile \
            --output /home/zhanghaoye/data/eval_test/$REPLY.llava_test_gpt4.jsonl &
        sleep 5

        C=$((C+1))
        echo "C=$C"
        if [[ $C == $process_limit ]]; then
            echo "Wait for next iteration"
            C=0
            wait
        fi
    fi
done 9< <( find $SOURCE_DIR -type f -name "*qa*" -exec printf '%s\0' {} + )

wait