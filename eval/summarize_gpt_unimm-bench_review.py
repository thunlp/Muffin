import json
import os
from collections import defaultdict

import numpy as np
import sys
import glob


if __name__ == '__main__':
    base_dir = sys.argv[1]
    print(base_dir)

    patterns = ['*', '*/*', '*/*/*']
    f_list = sum([list(glob.glob(os.path.join(base_dir, p))) for p in patterns], [])
    review_files = [x for x in f_list if x.endswith('.jsonl') and 'unimm-bench_gpt4_eval' in x and 'gpt4' in x]

    for review_file in sorted(review_files):
        config = review_file.replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(f'{config} #{len(list(open(review_file)))}')
        with open(review_file) as f:
            for review_str in f:
                try:
                    review = json.loads(review_str)
                    # filter failed case
                    if review['content'].startswith('-1\n'):
                        print(f'#### Skip fail Case')
                        continue
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                except:
                    print(f'Error parsing {review_str}')
        for k, v in scores.items():
            mean = np.asarray(v).mean() / 5 * 100
            print(f'{k:<7s} {mean: .2f}')
        print('=================================')
