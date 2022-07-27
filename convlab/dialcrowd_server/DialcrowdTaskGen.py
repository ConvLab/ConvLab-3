'''
DialcrowdTaskGen.py - Generating tasks for Dialcrowd.
==========================================================================

Only work for MultiWoz domain.

@author: Songbo

'''

import json
import argparse
from uuid import uuid1 as uuid
from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.policy.tus.multiwoz.Goal import Goal


def timeline_task(goal, task_id, goal_generator):
    goal["taskID"] = task_id
    # TODO need to modify the Dialcrowd server backend
    goal["taskType"] = "convlab2"
    goal["taskMessage"] = goal_generator.build_message(goal)
    return goal


def multiwoz_task(goal, task_id):
    task_info = {}
    task_info["taskID"] = task_id
    task_info["tasks"] = []
    for domain in goal:
        task = {
            'reqs': goal[domain].get('reqt', []),
        }
        reqs = goal[domain].get('reqt', [])

        for slot_type in ['info', 'book']:
            task[slot_type] = slot_str(goal, domain, slot_type)

        task_info["tasks"].append({
            "Dom": domain.capitalize(),
            "Cons": ", ".join(task['info']),
            "Book": ", ".join(task['book']),
            "Reqs": ", ".join(task['reqs'])})

    return task_info


def normal_task(goal, task_id):
    task_info = {}
    task_info["taskID"] = task_id
    task_info["tasks"] = []
    for domain in goal["domain_ordering"]:
        task = {
            'reqs': goal[domain].get('reqt', []),
        }
        reqs = goal[domain].get('reqt', [])

        for slot_type in ['info', 'book']:
            task[slot_type] = slot_str(goal, domain, slot_type)

        task_info["tasks"].append({
            "Dom": domain.capitalize(),
            "Cons": ", ".join(task['info']),
            "Book": ", ".join(task['book']),
            "Reqs": ", ".join(task['reqs'])})

    return task_info


def slot_str(goal, domain, slot_type):
    slots = []
    for slot in goal[domain].get(slot_type, []):
        value = info_str(goal, domain, slot, slot_type)
        slots.append(f"{slot}={value}")
    return slots


def check_slot_length(task_info, max_slot_len=5):
    for task in task_info["tasks"]:
        if len(task["Cons"].split(", ")) > max_slot_len:
            print(f'SLOT LENGTH ERROR: {len(task["Cons"].split(", "))}')
            return False
    return True


def check_domain_number(task_info, min_domain_len=0, max_domain_len=3):
    if len(task_info["tasks"]) > min_domain_len and len(task_info["tasks"]) <= max_domain_len:
        return True
    print(f'DOMAIN NUMBER ERROR: {len(task_info["tasks"])}')
    return False


def info_str(goal, domain, slot, slot_type):
    if slot_type == 'info':
        fail_type = 'fail_info'
    elif slot_type == 'book':
        fail_type = 'book_again'
    # print(goal, domain, slot, slot_type)
    value = goal[domain][slot_type][slot]
    if fail_type not in goal[domain]:
        return value
    else:
        fail_info = goal[domain][fail_type].get(slot, "")
        if fail_info and fail_info != value:
            return f"{fail_info} (if unavailable use: {value})"
        else:
            return value


def write_task(task_size, task_type, out_file, test_file=None):
    goal_generator = GoalGenerator(boldify=True)
    if test_file:
        test_list = [task_id for task_id in test_file]
    con = 0
    output = []
    while len(output) < task_size:
        # try:
        if test_file:
            goal = Goal(goal=test_file[test_list[len(output)]]["goal"])
            task_info = multiwoz_task(goal.domain_goals, len(output) + 10000)

        else:
            goal = goal_generator.get_user_goal()
            if 'police' in goal['domain_ordering']:
                no_police = list(goal['domain_ordering'])
                no_police.remove('police')
                goal['domain_ordering'] = tuple(no_police)
                del goal['police']

            if task_type == "convlab2":
                task_info = timeline_task(
                    goal, len(output) + 10000, goal_generator)
            elif task_type == "normal":
                # task_info = normal_task(goal, len(output) + 10000)
                task_info = normal_task(goal, str(uuid()))
            else:
                print("unseen task type. No goal is created.")

        if check_domain_number(task_info) and check_slot_length(task_info):
            output.append(json.dumps(task_info))

        # except Exception as e:
        #     print(goal)
        #     con += 1

    print("{} exceptions in total." .format(con))
    f = open(out_file, "w")
    f.write("\n".join(output))
    f.close()


def write_task_single_domain(task_size, task_type, out_file, test_file=None):
    goal_generator = GoalGenerator(boldify=True)
    if test_file:
        test_list = [task_id for task_id in test_file]
    con = 0
    output = []
    goals_per_domain = dict()
    # we want task_size / 5 many goals for each of the 5 domains we have
    goals_needed = task_size / 5
    while len(output) < task_size:
        # try:
        if test_file:
            goal = Goal(goal=test_file[test_list[len(output)]]["goal"])
            task_info = multiwoz_task(goal.domain_goals, len(output) + 10000)

        else:
            goal = goal_generator.get_user_goal()
            if 'police' in goal['domain_ordering']:
                no_police = list(goal['domain_ordering'])
                no_police.remove('police')
                goal['domain_ordering'] = tuple(no_police)
                del goal['police']
            if 'hospital' in goal['domain_ordering']:
                no_police = list(goal['domain_ordering'])
                no_police.remove('hospital')
                goal['domain_ordering'] = tuple(no_police)
                del goal['hospital']

            # make sure we only get single domain goals
            num_goals = len(goal['domain_ordering'])

            if num_goals == 0:
                continue

            while num_goals > 1:
                domain_removed = list(goal['domain_ordering'])[-1]
                goal['domain_ordering'] = tuple(list(goal['domain_ordering'])[:-1])
                del goal[domain_removed]
                num_goals = len(goal['domain_ordering'])

            domain = goal['domain_ordering']

            type_request = False
            for key in goal:
                if key == "domain_ordering":
                    continue
                if 'reqt' in goal[key]:
                    if 'type' in goal[key]['reqt']:
                        type_request = True
                        break
            if type_request:
                continue

            # if we have enough domains, continue to search
            if goals_per_domain.get(domain, 0) >= goals_needed:
                continue

            if domain not in goals_per_domain:
                goals_per_domain[domain] = 1
            else:
                goals_per_domain[domain] += 1

            if task_type == "convlab2":
                task_info = timeline_task(
                    goal, len(output) + 10000, goal_generator)
            elif task_type == "normal":
                task_info = normal_task(goal, str(uuid()))
            else:
                print("unseen task type. No goal is created.")

        if check_domain_number(task_info) and check_slot_length(task_info):
            output.append(json.dumps(task_info))

        # except Exception as e:
        #     print(goal)
        #     con += 1

    print("{} exceptions in total." .format(con))
    f = open(out_file, "w")
    f.write("\n".join(output))
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-task', default=400, type=int,
                        help="How many tasks would you like in you task list.")
    parser.add_argument('--task-type', default="normal", type=str,
                        help="the format of task type, 'normal' or 'timeline'.")
    parser.add_argument('--out-file', default="task.out", type=str,
                        help="the output file name")
    parser.add_argument('--test-file', default="", type=str,
                        help="the multiwoz test file")
    parser.add_argument('--single-domains', action='store_true',
                        help="if it should generate uniform distribution over single domain goals")

    args = parser.parse_args()
    test_file = None
    num_task = args.num_task
    if args.test_file:
        test_file = json.load(open(args.test_file))
        num_task = len(test_file)
        print(f"use test file, length={num_task}")
    if not args.single_domains:
        write_task(num_task, args.task_type, args.out_file, test_file)
    else:
        write_task_single_domain(num_task, args.task_type, args.out_file, test_file)


if __name__ == '__main__':

    # How many tasks would you like in you task list.
    main()


# END OF FILE
