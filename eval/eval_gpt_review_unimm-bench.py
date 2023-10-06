import argparse
import json
import os

import tqdm
import time
import pathlib

from gpt4_grpc import get_eval


def avg(lst):
    return sum(lst) / len(lst)


def parse_score(review):
    try:
        score_str = review.split('\n')[0]
        return float(score_str)
    except Exception as e:
        print(e)
        print('error', review)
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-a', '--answer')
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--limit', default=100, type=int)
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    questions = json.load(f_q)
    category_lst = ["AOKVQA", "GQA", "OKVQA", "VQAv2"]
    for i, ques_js in enumerate(questions):
        ques_js["category"] = category_lst[i // 100]
    ans1_list = [json.loads(line) for line in open(os.path.expanduser(args.answer))]

    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    reviewed_lines = []
    score_list = []
    if pathlib.Path(args.output).exists():
        reviewed_lines = open(args.output).readlines()[:-1]
        print(f'Resume {args.output} from {len(reviewed_lines)}')
    review_file = open(f'{args.output}', 'w')
    if reviewed_lines:
        for line in reviewed_lines:
            review_file.write(line)
            score_list.append(json.loads(line)['tuple'])
            review_file.flush()

    chat = 'gpt-4-0314'

    js_list = []
    handles = []
    category_sample_count = dict()

    for line_idx, (ques_js, ans1_js) in tqdm.tqdm(enumerate(zip(questions, ans1_list)), total=400):
        if line_idx < len(reviewed_lines):
            continue
        category = ques_js['category']

        if category not in category_sample_count:
            category_sample_count[category] = 0
        if category_sample_count[category] == args.limit:
            continue
        category_sample_count[category] += 1
        ques, ans1 = ques_js, ans1_js

        rule = rule_dict["vqa_standard"]
        prompt = rule['prompt']
        role = rule['role']

        content = (f'[Question]\n<image>\n{ques["question"]}\n\n'
                   f'[{role} Response]\n{ans1["text"]}\n\n[End of {role} Response]\n\n'
                   f'[System]\n{prompt}\n\n'
                   f'[Expected Answer]\n{ques["answer"]}\n\n')

        if category == "AOKVQA":
            content = content + (f'[Rationale to Get Answer]\n{ques["context"]}\n\n'
                                 f'[Human Answers]\n{ques["Human Answers"].replace("[", " ").replace("]", "")}\n\n'
                                 )
        elif category == "GQA":
            content = content + (f'[Human Answers]\n{ques["Human Answers"]}\n\n'
                                 )
        elif category == "OKVQA":
            human_answers = ques['Human Answers'].replace(
                ']', '').replace('Human answers are: [', '[Human Answers]\n ')
            content = content + (f'{human_answers}\n\n'
                                 )
        elif category == "ScienceQA":
            content = content + (f'[Hint]\n{ques["context"]}\n\n'
                                 f'[Human Answers]\n{ques["Human Answers"]}\n\n'
                                 )
        elif category == "VizWiz":
            human_answers = ques['Human Answers'].replace(
                ']', '').replace('Human answers are: [', '[Human Answers]\n ')
            content = content + (f'[Image Description]\n{ques["context"]}\n\n'
                                 f'{human_answers}\n\n'
                                 )
        elif category == "VQAv2":
            content = content + (f'[Human Answers]\n{ques["Human Answers"].replace("[", " ").replace("]", "")}\n\n'
                                 )
        elif category == "WebQA":
            content = content + (f'[Webpage Title]\n{ques["context"]}\n\n'
                                 f'[Human Answers]\n{ques["answer"]}\n\n'
                                 )
        output = {
            'id': line_idx + 1,
            'question_id': ans1.get('question_id', line_idx + 1),
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            # 'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category,
            'input_content': content}

        review = get_eval(chat, content)

        interval = len(content.split()) // 50
        time.sleep(interval)

        score = parse_score(review)
        if score == -1:
            start = len(score_list) // args.limit * args.limit
            score = avg(score_list[start:])
            print(f'Drop One Sample, use smoothed value of #{start} to #{len(score_list)} = {score:.2f}')

        score_list.append(score)

        output['content'] = review
        output['tuple'] = score
        review_file.write(json.dumps(output) + '\n')
        review_file.flush()

    review_file.close()
