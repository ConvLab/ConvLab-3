import json
import numpy as np

from convlab.evaluator.multiwoz_eval import MultiWozEvaluator

old_multiwoz_path = "/Users/geishaus/Downloads/MultiWOZ_2.1/data.json"


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_goal(goal):
    processed_goal = {}
    for key, value in goal.items():
        if value and key != "message" and key != "topic":
            if 'reqt' in value:
                value['reqt'] = dict((r, "?") for r in value['reqt'])
            processed_goal[key] = value
    return processed_goal

def load_multiwoz_goals(file_path):
    multiwoz = load_json(file_path)
    goals = dict((key.split(".json")[0], process_goal(value["goal"])) for key, value in multiwoz.items())
    return goals

def add_turn(evaluator, turn):
    user_action = turn["state"]["user_action"]
    sys_act = turn["sys_act"]

    evaluator.add_usr_da(user_action)
    evaluator.add_sys_da(sys_act)

def calculate_success_inform(evaluator, soft=False):
    prec, rec, F1 = evaluator.inform_F1()
    if not soft:
        success = 1 if (rec == 1 or rec is None) else 0
        inform = 1 if evaluator.final_goal_analyze() == 1 else 0
    else:
        success = rec if rec is not None else 1
        inform = evaluator.final_goal_analyze()
    return success, inform

def create_dialogue_dicts(file_path, soft=False):

    corpus_logs = load_json(file_path)
    dialogue_dicts = []
    evaluator = MultiWozEvaluator()
    goals = load_multiwoz_goals(old_multiwoz_path)

    dialogue_id = corpus_logs[0]["utt_idx"].split(".json")[0]
    evaluator.add_goal(goals[dialogue_id])

    for i, turn in enumerate(corpus_logs):

        dialogue_utt_id = turn["utt_idx"].split(".json")[0]

        if dialogue_utt_id != dialogue_id:
            # dialogue has finished and we need to calculate inform and success
            success, inform = calculate_success_inform(evaluator, soft=soft)
            dialogue_dicts.append({"dialogue_id": dialogue_id, "inform": inform, "success": success})

            # reset evaluator
            dialogue_id = dialogue_utt_id
            goal = goals[dialogue_id]
            evaluator.add_goal(goal)
            add_turn(evaluator, turn)

        else:
            add_turn(evaluator, turn)

    return dialogue_dicts


if __name__ == "__main__":
    file_path = "results.json"
    dialogue_dicts = create_dialogue_dicts(file_path, soft=False)

    success_list = [dialogue["success"] for dialogue in dialogue_dicts]
    inform_list = [dialogue["inform"] for dialogue in dialogue_dicts]

    success_rate = np.mean(success_list)
    inform_rate = np.mean(inform_list)

    print(f"Success rate: {success_rate}, Inform rate: {inform_rate}")