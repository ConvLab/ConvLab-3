import json
import os
from pprint import pprint

import numpy as np
from scipy import stats

emotion_dict = {"satisfied": 1, "neutral": 0, 'excited': 1, 'fearful': 0, 'apologetic': 0,
                    "dissatisfied": -1, "abusive": -1}

model_names = ['simpleemollama', 'simpleconductllama']
seeds = [0, 1, 2, 3, 4]

log_path_prefix = 'path_to/results'

E = [[], []]
S = [[], []]
SEEDS = []
for s in seeds:
    emo = []
    suc = []
    for i, model_name in enumerate(model_names):
        emo_scores = []
        suc_scores = []
        log_path = os.path.join(log_path_prefix, f'{model_name}_{str(s)}', 'conversation')
        try:
            conv_json_name = os.listdir(log_path)[0]
            conv_json_path = os.path.join(log_path, conv_json_name)
            with open(conv_json_path, 'r') as f:
                log = json.load(f)
        except:
            continue

        for conv in log['conversation']:
            conv_emo_scores = []
            suc_scores.append(conv['Success'])
            for turn in conv['log']:
                if turn['role'] == 'usr':
                    emotion = turn['emotion'].lower()
                    emo_score = emotion_dict[emotion]
                    conv_emo_scores.append(emo_score)
            emo_scores.append(np.mean(conv_emo_scores))
        print(np.mean(emo_scores))
        # print(np.mean(suc_scores))
        emo.append(np.mean(emo_scores))
        suc.append(np.mean(suc_scores))
        E[i] += emo_scores
        S[i] += suc_scores
    E.append(emo_scores)
    S.append(suc_scores)

    if emo[1] > 0.2 and emo[0]> 0.2:
        SEEDS.append(s)

print(np.mean(E[0]), np.mean(E[1]))
print(stats.ttest_ind(E[0], E[1]))
print(np.mean(S[0]), np.mean(S[1]))
print(stats.ttest_ind(S[0], S[1]))
print(SEEDS)