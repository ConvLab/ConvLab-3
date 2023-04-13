import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_dir = "convlab/policy/emoUS/result"


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, help="the conversation file")
    return parser.parse_args()


def basic_analysis(conversation):
    info = {"Complete": [], "Success": [], "Success strict": [], "turns": []}
    for dialog in conversation:
        for x in info:
            info[x].append(dialog[x])
    for x in info:
        info[x] = np.mean(info[x])
    return info


def advance(conversation):
    info = {}
    for dialog in conversation:
        temp = turn_level(dialog["log"])
        for metric, data in temp.items():
            if metric not in info:
                info[metric] = {}
            for emotion, count in data.items():
                if emotion not in info[metric]:
                    info[metric][emotion] = 0
                info[metric][emotion] += count

    return info


def get_turn_emotion(conversation):
    """ Get the emotion of each turn in the conversation 
    Args:
        conversation (list): a list of dialog
    Returns:
        turn_emotion (list): a list of emotion of each turn
    """
    turn_info = {"all": {},
                 "Complete": {}, "Not Complete": {},
                 "Success": {}, "Not Success": {},
                 "Success strict": {}, "Not Success strict": {}}
    max_turn = 0
    for dialog in conversation:
        for i in range(0, len(dialog["log"]), 2):
            turn = int(i / 2)
            if turn > max_turn:
                max_turn = turn
            emotion = emotion_score(dialog["log"][i]["emotion"])
            insert_turn(turn_info["all"], turn, emotion)
            for metric in ["Complete", "Success", "Success strict"]:
                if dialog[metric]:
                    insert_turn(turn_info[metric], turn, emotion)
                else:
                    insert_turn(turn_info[f"Not {metric}"], turn, emotion)
    print("MAX_TURN", max_turn)
    data = {'x': [t for t in range(max_turn)],
            'all_positive': [],
            'all_negative': [],
            'all_mean': [],
            'all_std': []}

    for metric in ["Complete", "Success", "Success strict"]:
        data[f"{metric}_positive"] = []
        data[f"{metric}_negative"] = []
        data[f"{metric}_mean"] = []
        data[f"{metric}_std"] = []
        data[f"Not {metric}_positive"] = []
        data[f"Not {metric}_negative"] = []
        data[f"Not {metric}_mean"] = []
        data[f"Not {metric}_std"] = []

    for t in range(turn):
        pos, neg, mean, std = turn_score(turn_info["all"][t])
        data[f"all_positive"].append(pos)
        data[f"all_negative"].append(neg)
        data[f"all_mean"].append(mean)
        data[f"all_std"].append(std)
        for raw_metric in ["Complete", "Success", "Success strict"]:
            for metric in [raw_metric, f"Not {raw_metric}"]:
                if t not in turn_info[metric]:
                    data[f"{metric}_positive"].append(0)
                    data[f"{metric}_negative"].append(0)
                    data[f"{metric}_mean"].append(0)
                    data[f"{metric}_std"].append(0)
                else:
                    pos, neg, mean, std = turn_score(turn_info[metric][t])
                    data[f"{metric}_positive"].append(pos)
                    data[f"{metric}_negative"].append(neg)
                    data[f"{metric}_mean"].append(mean)
                    data[f"{metric}_std"].append(std)
    for x in data:
        data[x] = np.array(data[x])

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    p = {"Complete": {"color": "C0", "label": "Success"},
         "Not Complete": {"color": "C1", "label": "Fail"},
         "all": {"color": "C2", "label": "all"}}
    for name, para in p.items():

        ax.plot(data['x'],
                data[f"{name}_mean"],
                'o--',
                color=para["color"],
                label=para["label"])
        ax.fill_between(data['x'],
                        data[f"{name}_mean"]+data[f"{name}_std"],
                        data[f"{name}_mean"]-data[f"{name}_std"],
                        color=para["color"], alpha=0.2)

    ax.legend()
    ax.set_xlabel("turn")
    ax.set_ylabel("Sentiment")
    ax.set_xticks([t for t in range(0, max_turn, 2)])
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "turn2emotion.png"))


def turn_score(score_list):
    count = len(score_list)
    positive = 0
    negative = 0
    for s in score_list:
        if s > 0:
            positive += 1
        if s < 0:
            negative += -1
    return positive/count, negative/count, np.mean(score_list), np.std(score_list, ddof=1)/np.sqrt(len(score_list))


def insert_turn(turn_info, turn, emotion):
    if turn not in turn_info:
        turn_info[turn] = []
    turn_info[turn].append(emotion)


def emotion_score(emotion):
    if emotion == "Neutral":
        return 0
    if emotion in ["Satisfied", "Excited"]:
        return 1
    return -1


def plot(conversation):
    pass


def turn_level(dialog):
    # metric: {emotion: count}
    dialog_info = {}
    for index in range(2, len(dialog), 2):
        pre_usr = dialog[index-2]
        sys = dialog[index-1]
        cur_usr = dialog[index]
        info = neglect_reply(pre_usr, sys, cur_usr)
        append_info(dialog_info, info)
        info = confirm(pre_usr, sys, cur_usr)
        append_info(dialog_info, info)
        info = miss_info(pre_usr, sys, cur_usr)
        append_info(dialog_info, info)
        if index > 2:
            info = loop(dialog[index-3], sys, cur_usr)
            append_info(dialog_info, info)

    return dialog_info

# provide wrong info
# action length
# incomplete info?


def append_info(dialog_info, info):
    if not info:
        return
    for emotion, metric in info.items():
        if metric not in dialog_info:
            dialog_info[metric] = {}
        if emotion not in dialog_info[metric]:
            dialog_info[metric][emotion] = 0
        dialog_info[metric][emotion] += 1


def get_inform(act):
    inform = {}
    for intent, domain, slot, value in act:
        if intent not in ["inform", "recommend"]:
            continue
        if domain not in inform:
            inform[domain] = []
        inform[domain].append(slot)
    return inform


def get_request(act):
    request = {}
    for intent, domain, slot, _ in act:
        if intent == "request":
            if domain not in request:
                request[domain] = []
            request[domain].append(slot)
    return request


def neglect_reply(pre_usr, sys, cur_usr):
    request = get_request(pre_usr["act"])
    if not request:
        return {}

    system_inform = get_inform(sys["utt"])

    for domain, slots in request.items():
        if domain not in system_inform:
            return {cur_usr["emotion"]: "neglect"}
        for slot in slots:
            if slot not in system_inform[domain]:
                return {cur_usr["emotion"]: "neglect"}
    return {cur_usr["emotion"]: "reply"}


def miss_info(pre_usr, sys, cur_usr):
    system_request = get_request(sys["utt"])
    if not system_request:
        return {}
    user_inform = get_inform(pre_usr["act"])
    for domain, slots in system_request.items():
        if domain not in user_inform:
            continue
        for slot in slots:
            if slot in user_inform[domain]:
                return {cur_usr["emotion"]: "miss_info"}
    return {}


def confirm(pre_usr, sys, cur_usr):
    user_inform = get_inform(pre_usr["act"])

    if not user_inform:
        return {}

    system_inform = get_inform(sys["utt"])

    for domain, slots in user_inform.items():
        if domain not in system_inform:
            continue
        for slot in slots:
            if slot in system_inform[domain]:
                return {cur_usr["emotion"]: "confirm"}

    return {cur_usr["emotion"]: "no confirm"}


def loop(s0, s1, u1):
    if s0 == s1:
        return {u1["emotion"]: "loop"}


def dict2csv(data):
    r = {}
    emotion = json.load(open("convlab/policy/emoUS/emotion.json"))
    for act, value in data.items():
        temp = [0]*(len(emotion)+1)
        for emo, count in value.items():
            temp[emotion[emo]] = count
        temp[-1] = sum(temp)
        for i in range(len(emotion)):
            temp[i] /= temp[-1]
        r[act] = temp
    dataframe = pd.DataFrame.from_dict(
        r, orient='index', columns=[emo for emo in emotion]+["count"])
    dataframe.to_csv(open(os.path.join(result_dir, "act2emotion.csv"), 'w'))


def main():
    args = arg_parser()
    result = {}
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    conversation = json.load(open(args.file))["conversation"]
    # basic_info = basic_analysis(conversation)
    # result["basic_info"] = basic_info
    # print(basic_info)
    # advance_info = advance(conversation)
    # print(advance_info)
    # result["advance_info"] = advance_info
    # json.dump(result, open(
    #     os.path.join("conversation_result.json"), 'w'), indent=2)
    # dict2csv(advance_info)
    get_turn_emotion(conversation)


if __name__ == "__main__":
    main()
