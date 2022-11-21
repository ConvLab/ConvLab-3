import sys
import os
import logging
import random
import time
import json
import zipfile
import numpy as np
import torch
from tensorboardX import SummaryWriter
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.file_util import cached_path
from convlab.policy.evaluate_distributed import evaluate_distributed
from convlab.util.train_util_neo import init_logging_nunu
from convlab.dialog_agent.agent import PipelineAgent
from convlab.dialog_agent.session import BiSession
from convlab.dialog_agent.env import Environment
from convlab.dst.rule.multiwoz import RuleDST
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.util import load_dataset

import shutil
import signal


slot_mapping = {"pricerange": "price range", "post": "postcode", "arriveBy": "arrive by", "leaveAt": "leave at",
                "Id": "trainid", "ref": "reference"}


sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE


class timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def move_finished_training(dir_in, dir_to):
    os.makedirs(dir_to, exist_ok=True)
    shutil.move(dir_in, dir_to)
    logging.info("Moved results to finished experiments folder.")


def flatten_acts(dialogue_acts):
    act_list = []
    for act_type in dialogue_acts:
        for act in dialogue_acts[act_type]:
            act_list.append([act['intent'], act['domain'],
                            act['slot'], act.get('value', "")])
    return act_list


def load_config_file(filepath: str = None) -> dict:
    """
    load config setting from json file
    :param filepath: str, dest config file path
    :return: dict,
    """
    if not isinstance(filepath, str):
        filepath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'envs_config.json'))
    # load
    with open(filepath, 'r', encoding='UTF-8') as f:
        conf = json.load(f)
    assert isinstance(
        conf, dict), 'Incorrect format in config file \'%s\'' % filepath

    # check sections
    for sec in ['model', 'vectorizer_sys', 'nlu_sys', 'dst_sys', 'sys_nlg', 'nlu_usr', 'policy_usr', 'usr_nlg']:
        assert sec in conf.keys(), 'Missing \'%s\' section in config file \'%s\'' % (sec, filepath)

    return conf


def save_config(terminal_args, config_file_args, config_save_path, policy_config=None):
    config_save_path = os.path.join(config_save_path, f'config_saved.json')
    args_dict = {"args": terminal_args, "config": config_file_args, "policy_config": policy_config}
    json.dump(args_dict, open(config_save_path, 'w'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logging(root_dir, mode):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    dir_path = os.path.join(root_dir, f'experiments/experiment_{current_time}')
    # init_logging_nunu(dir_path)
    _, log_save_path = init_logging_nunu(dir_path, mode)
    save_path = os.path.join(dir_path, 'save')
    os.makedirs(save_path, exist_ok=True)
    config_save_path = os.path.join(dir_path, 'configs')
    os.makedirs(config_save_path, exist_ok=True)
    logger = logging.getLogger()
    logging.info(f"Visible device: {device}")
    tb_writer = SummaryWriter(os.path.join(dir_path, f'TB_summary'))
    return logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path


def log_start_args(conf):
    logging.info(f"Epochs to train: {conf['model']['epoch']}")
    logging.info(f"Eval-frequency: {conf['model']['eval_frequency']}")
    logging.info(f"Seed: {conf['model']['seed']}")
    logging.info(
        f"We use {conf['model']['num_eval_dialogues']} dialogues for evaluation.")


def save_best(policy_sys, best_complete_rate, best_success_rate, best_return, complete_rate, success_rate, avg_return,
              save_path):
    # policy_sys.save(save_path, "best")
    if avg_return > best_return:
        logging.info("Saving best policy.")
        policy_sys.save(save_path, "best")
        best_return = avg_return
    if success_rate > best_success_rate:
        best_success_rate = success_rate
    if complete_rate > best_complete_rate:
        best_complete_rate = complete_rate
        # policy_sys.save(save_path, "best")
    logging.info(
        f"Best Complete Rate: {best_complete_rate}, Best Success Rate: {best_success_rate}, "
        f"Best Average Return: {best_return}")
    return best_complete_rate, best_success_rate, best_return


def eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path, single_domain_goals=False, allowed_domains=None):
    policy_sys.is_train = False

    goal_generator = GoalGenerator()
    goals = []
    for seed in range(1000, 1000 + conf['model']['num_eval_dialogues']):
        set_seed(seed)
        goal = create_goals(goal_generator, 1, single_domain_goals, allowed_domains)
        goals.append(goal[0])

    if conf['model']['process_num'] == 1:
        complete_rate, success_rate, success_rate_strict, avg_return, turns, \
            avg_actions, task_success, book_acts, inform_acts, request_acts, \
                select_acts, offer_acts, recommend_acts = evaluate(sess,
                                                num_dialogues=conf['model']['num_eval_dialogues'],
                                                sys_semantic_to_usr=conf['model'][
                                                    'sys_semantic_to_usr'],
                                                save_flag=save_eval, save_path=log_save_path, goals=goals)

        total_acts = book_acts + inform_acts + request_acts + select_acts + offer_acts + recommend_acts
    else:
        complete_rate, success_rate, success_rate_strict, avg_return, turns, \
            avg_actions, task_success, book_acts, inform_acts, request_acts, \
            select_acts, offer_acts, recommend_acts = \
            evaluate_distributed(sess, list(range(1000, 1000 + conf['model']['num_eval_dialogues'])),
                                 conf['model']['process_num'], goals)
        total_acts = book_acts + inform_acts + request_acts + select_acts + offer_acts + recommend_acts

        task_success_gathered = {}
        for task_dict in task_success:
            for key, value in task_dict.items():
                if key not in task_success_gathered:
                    task_success_gathered[key] = []
                task_success_gathered[key].append(value)
        task_success = task_success_gathered

    policy_sys.is_train = True

    mean_complete, err_complete = np.average(complete_rate), np.std(complete_rate) / np.sqrt(len(complete_rate))
    mean_success, err_success = np.average(success_rate), np.std(success_rate) / np.sqrt(len(success_rate))
    mean_success_strict, err_success_strict = np.average(success_rate_strict), np.std(success_rate_strict) / np.sqrt(len(success_rate_strict))
    mean_return, err_return = np.average(avg_return), np.std(avg_return) / np.sqrt(len(avg_return))
    mean_turns, err_turns = np.average(turns), np.std(turns) / np.sqrt(len(turns))
    mean_actions, err_actions = np.average(avg_actions), np.std(avg_actions) / np.sqrt(len(avg_actions))

    logging.info(f"Complete: {mean_complete}+-{round(err_complete, 2)}, "
                 f"Success: {mean_success}+-{round(err_success, 2)}, "
                 f"Success strict: {mean_success_strict}+-{round(err_success_strict, 2)}, "
                 f"Average Return: {mean_return}+-{round(err_return, 2)}, "
                 f"Turns: {mean_turns}+-{round(err_turns, 2)}, "
                 f"Average Actions: {mean_actions}+-{round(err_actions, 2)}, "
                 f"Book Actions: {book_acts/total_acts}, Inform Actions: {inform_acts/total_acts}, "
                 f"Request Actions: {request_acts/total_acts}, Select Actions: {select_acts/total_acts}, "
                 f"Offer Actions: {offer_acts/total_acts}, Recommend Actions: {recommend_acts/total_acts}")

    for key in task_success:
        logging.info(
            f"{key}: Num: {len(task_success[key])} Success: {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}")

    return {"complete_rate": mean_complete,
            "success_rate": mean_success,
            "success_rate_strict": mean_success_strict,
            "avg_return": mean_return,
            "turns": mean_turns,
            "avg_actions": mean_actions,
            "book_acts": book_acts/total_acts,
            "inform_acts": inform_acts/total_acts,
            "request_acts": request_acts/total_acts,
            "select_acts": select_acts/total_acts,
            "offer_acts": offer_acts/total_acts,
            "recommend_acts": recommend_acts/total_acts}


def env_config(conf, policy_sys, check_book_constraints=True):
    nlu_sys = conf['nlu_sys_activated']
    dst_sys = conf['dst_sys_activated']
    sys_nlg = conf['sys_nlg_activated']
    nlu_usr = conf['nlu_usr_activated']
    dst_usr = conf['dst_usr_activated']
    policy_usr = conf['policy_usr_activated']
    usr_nlg = conf['usr_nlg_activated']

    # Setup uncertainty thresholding
    if dst_sys:
        try:
            if dst_sys.return_confidence_scores:
                policy_sys.vector.setup_uncertain_query(dst_sys.confidence_thresholds)
        except:
            logging.info('Uncertainty threshold not set.')

    simulator = PipelineAgent(nlu_usr, dst_usr, policy_usr, usr_nlg, 'user')
    system_pipeline = PipelineAgent(nlu_sys, dst_sys, policy_sys, sys_nlg,
                                    'sys', return_semantic_acts=conf['model']['sys_semantic_to_usr'])

    # assemble
    evaluator = MultiWozEvaluator(
        check_book_constraints=check_book_constraints)
    env = Environment(sys_nlg, simulator, nlu_sys, dst_sys, evaluator=evaluator,
                      use_semantic_acts=conf['model']['sys_semantic_to_usr'])
    sess = BiSession(system_pipeline, simulator, None, evaluator)

    return env, sess


def create_env(args, policy_sys):
    if args.use_setsumbt_tracker:
        from convlab.nlu.jointBERT.multiwoz import BERTNLU
        from convlab.nlg.template.multiwoz import TemplateNLG
        from convlab.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
        nlu_sys = None
        dst_sys = SetSUMBTTracker(model_type='roberta',
                                  model_path=args.setsumbt_path,
                                  get_confidence_scores=args.use_confidence_scores,
                                  return_entropy=args.use_state_entropy,
                                  return_mutual_info=args.use_state_mutual_info)
        sys_nlg = TemplateNLG(is_user=False)

        nlu_usr = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                          model_file=args.nlu_model_path) if not args.sys_semantic_to_usr else None
        policy_usr = RulePolicy(character='usr')
        user_nlg = TemplateNLG(is_user=True, label_noise=args.user_label_noise,
                               text_noise=args.user_text_noise, seed=args.seed)

        simulator = PipelineAgent(nlu_usr, None, policy_usr, user_nlg, 'user')
        system_pipeline = PipelineAgent(nlu_sys, dst_sys, policy_sys, sys_nlg, 'sys',
                                        return_semantic_acts=args.sys_semantic_to_usr)
    elif args.use_bertnlu_rule_tracker:
        from convlab.nlu.jointBERT.multiwoz import BERTNLU
        from convlab.nlg.template.multiwoz import TemplateNLG
        nlu_sys = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                          model_file=args.nlu_model_path)
        dst_sys = RuleDST()
        sys_nlg = TemplateNLG(is_user=False)

        nlu_usr = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                          model_file=args.nlu_model_path) if not args.sys_semantic_to_usr else None
        policy_usr = RulePolicy(character='usr')
        user_nlg = TemplateNLG(
            is_user=True, label_noise=args.user_label_noise, text_noise=args.user_text_noise)

        simulator = PipelineAgent(nlu_usr, None, policy_usr, user_nlg, 'user')
        system_pipeline = PipelineAgent(nlu_sys, dst_sys, policy_sys, sys_nlg, 'sys',
                                        return_semantic_acts=args.sys_semantic_to_usr)
    else:
        nlu_sys = None
        dst_sys = RuleDST()
        sys_nlg = None
        policy_usr = RulePolicy(character='usr')

        simulator = PipelineAgent(None, None, policy_usr, None, 'user')
        system_pipeline = PipelineAgent(
            nlu_sys, dst_sys, policy_sys, sys_nlg, 'sys')

    # assemble
    evaluator = MultiWozEvaluator()
    env = Environment(sys_nlg, simulator, nlu_sys, dst_sys, evaluator=evaluator,
                      use_semantic_acts=args.sys_semantic_to_usr)
    sess = BiSession(system_pipeline, simulator, None, evaluator)

    return env, sess


def evaluate(sess, num_dialogues=400, sys_semantic_to_usr=False, save_flag=False, save_path=None, goals=None):

    eval_save = {}
    turn_counter_dict = {}
    turn_counter = 0.0

    task_success = {'All_user_sim': [], 'All_evaluator': [], "All_evaluator_strict": [],
                    'total_return': [], 'turns': [], 'avg_actions': [],
                    'total_booking_acts': [], 'total_inform_acts': [], 'total_request_acts': [],
                    'total_select_acts': [], 'total_offer_acts': [], 'total_recommend_acts': []}
    dial_count = 0
    for seed in range(1000, 1000 + num_dialogues):
        set_seed(seed)
        goal = goals.pop()
        sess.init_session(goal=goal)
        sys_response = [] if sess.sys_agent.nlg is None else ''
        sys_response = [] if sys_semantic_to_usr else sys_response
        avg_actions = 0
        total_return = 0.0
        turns = 0
        book = 0
        inform = 0
        request = 0
        select = 0
        offer = 0
        recommend = 0
        # this 40 represents the max turn of dialogue
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response)

            if len(sys_response) not in turn_counter_dict:
                turn_counter_dict[len(sys_response)] = 1
            else:
                turn_counter_dict[len(sys_response)] += 1

            acts = sess.sys_agent.dst.state['system_action']
            for intent, domain, _, _ in acts:
                if intent.lower() == 'book':
                    book += 1
                if intent.lower() == 'inform':
                    inform += 1
                if intent.lower() == 'request':
                    request += 1
                if intent.lower() == 'select':
                    select += 1
                if intent.lower() == 'offerbook':
                    offer += 1
                if intent.lower() == 'recommend':
                    recommend += 1
            avg_actions += len(acts)
            turn_counter += 1
            turns += 1
            total_return += reward

            if session_over is True:
                task_succ = sess.evaluator.task_success()
                complete = sess.evaluator.complete
                task_succ = sess.evaluator.success
                task_succ_strict = sess.evaluator.success_strict
                break
        else:
            complete = 0
            task_succ = 0
            task_succ_strict = 0

        for key in sess.evaluator.goal:
            if key not in task_success:
                task_success[key] = []
            else:
                task_success[key].append(task_succ_strict)

        task_success['All_user_sim'].append(complete)
        task_success['All_evaluator'].append(task_succ)
        task_success['All_evaluator_strict'].append(task_succ_strict)
        total_return = 80 if task_succ_strict else -40
        total_return -= turns
        task_success['total_return'].append(total_return)
        task_success['turns'].append(turns)
        task_success['avg_actions'].append(avg_actions / turns)

        task_success['total_booking_acts'].append(book)
        task_success['total_inform_acts'].append(inform)
        task_success['total_request_acts'].append(request)
        task_success['total_select_acts'].append(select)
        task_success['total_offer_acts'].append(offer)
        task_success['total_offer_acts'].append(offer)
        task_success['total_recommend_acts'].append(recommend)

        # print(agent_sys.agent_saves)
        eval_save['Conversation {}'.format(str(dial_count))] = [
            i for i in sess.sys_agent.agent_saves]
        sess.sys_agent.agent_saves.clear()
        dial_count += 1
        # print('length of dict ' + str(len(eval_save)))

    if save_flag:
        # print("what are you doing")
        save_file = open(os.path.join(save_path, 'evaluate_INFO.json'), 'w')
        json.dump(eval_save, save_file, cls=NumpyEncoder)
        save_file.close()
    # save dialogue_info and clear mem

    return np.average(task_success['All_user_sim']), np.average(task_success['All_evaluator']), \
        np.average(task_success['All_evaluator_strict']), np.average(task_success['total_return']), \
        np.average(task_success['turns']), np.average(task_success['avg_actions']), task_success, \
        np.average(task_success['total_booking_acts']), np.average(task_success['total_inform_acts']), \
        np.average(task_success['total_request_acts']), np.average(task_success['total_select_acts']), \
        np.average(task_success['total_offer_acts']), np.average(task_success['total_recommend_acts'])


def model_downloader(download_dir, model_path):
    """
    Function to download models from web url
    :param download_dir: Directory where models should be downloaded
    :param model_path: URL/Path of the model
    """
    logging.info('Load from model_file param')
    model_path = cached_path(model_path)
    archive = zipfile.ZipFile(model_path, 'r')
    archive.extractall(download_dir)
    archive.close()


def get_goal_distribution(dataset_name='multiwoz21'):

    data_split = load_dataset(dataset_name)
    domain_combinations = {}
    for key in data_split:
        data = data_split[key]
        for dialogue in data:
            goal = dialogue['goal']
            domains = list(set(goal['inform'].keys()) |
                           set(goal['request'].keys()))
            domains.sort()
            domains = "-".join(domains)

            if domains not in domain_combinations:
                domain_combinations[domains] = 1
            else:
                domain_combinations[domains] += 1

    single_domain_counter = {}
    for combi in domain_combinations:
        for domain in combi.split("-"):
            if domain not in single_domain_counter:
                single_domain_counter[domain] = domain_combinations[combi]
            else:
                single_domain_counter[domain] += domain_combinations[combi]

    domain_combinations = list(domain_combinations.items())
    domain_combinations = [list(pair) for pair in domain_combinations]
    domain_combinations.sort(key=lambda x: x[1], reverse=True)
    print(domain_combinations)
    print(single_domain_counter)
    print("Number of combinations:", sum(
        [value for _, value in domain_combinations]))


def unified_format(acts):
    new_acts = {'categorical': []}
    for act in acts:
        intent, domain, slot, value = act
        new_acts['categorical'].append(
            {"intent": intent, "domain": domain, "slot": slot, "value": value})

    return new_acts


def act_dict_to_flat_tuple(acts):
    tuples = []
    for domain_intent, svs in acts.items():
        for slot, value in svs:
            domain, intent = domain_intent.split('-')
            tuples.append([intent, domain, slot, value])


def create_goals(goal_generator, num_goals, single_domains=False, allowed_domains=None):
    from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal

    collected_goals = []
    while len(collected_goals) != num_goals:
        goal = Goal(goal_generator)
        if single_domains and len(goal.domain_goals) > 1:
            continue
        if allowed_domains is not None and not set(goal.domain_goals).issubset(set(allowed_domains)):
            continue
        collected_goals.append(goal)
    return collected_goals


def build_domains_goal(goal_generator, domains=None):
    from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
    found = False
    while not found:
        goal = Goal(goal_generator)
        if domains is None:
            found = True
        if set(goal.domain_goals) == domains:
            found = True
    return goal


def data_goals(num_goals, dataset="multiwoz21", dial_ids_order=0):
    from convlab.policy.tus.unify.Goal import Goal
    from convlab.policy.tus.unify.util import create_goal
    data = load_dataset(dataset, dial_ids_order)
    collected_goals = []
    for dialog in data["test"]:
        goal = Goal(create_goal(dialog))
        collected_goals.append(goal)
    if len(collected_goals) < num_goals:
        print(f"# of data goals ({data['test']}) < num_goals {num_goals}")
    # reorder goals?
    return collected_goals


def map_class(cls_path: str):
    """
    Map to class via package text path
    :param cls_path: str, path with `convlab` project directory as relative path, separator with `,`
                            E.g  `convlab.nlu.svm.camrest.nlu.SVMNLU`
    :return: class
    """
    pkgs = cls_path.split('.')
    cls = __import__('.'.join(pkgs[:-1]))
    for pkg in pkgs[1:]:
        cls = getattr(cls, pkg)
    return cls


def get_config(filepath, args) -> dict:
    """
    The configuration file is used to create all the information needed for the deployment,
    and the necessary security monitoring has been performed, including the mapping of the class.
    :param filepath: str, dest config file path
    :return: dict
    """

    conf = load_config_file(filepath)

    # add project root dir
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir)))

    for arg in args:
        if len(arg) == 3:
            conf[arg[0]][arg[1]] = arg[2]
        if len(arg) == 4:
            conf[arg[0]][arg[1]][arg[2]] = arg[3]
        if len(arg) == 5:
            conf[arg[0]][arg[1]][arg[2]][arg[3]] = arg[4]

    # Autoload uncertainty settings from policy based on the tracker used
    dst_name = [model for model in conf['dst_sys']]
    dst_name = dst_name[0] if dst_name else None
    vec_name = [model for model in conf['vectorizer_sys']]
    vec_name = vec_name[0] if vec_name else None
    if dst_name and 'setsumbt' in dst_name.lower():
        if 'return_confidence_scores' in conf['dst_sys'][dst_name]['ini_params']:
            param = conf['dst_sys'][dst_name]['ini_params']['return_confidence_scores']
            conf['vectorizer_sys'][vec_name]['ini_params']['use_confidence_scores'] = param
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_confidence_scores'] = False
        if 'return_belief_state_mutual_info' in conf['dst_sys'][dst_name]['ini_params']:
            param = conf['dst_sys'][dst_name]['ini_params']['return_belief_state_mutual_info']
            conf['vectorizer_sys'][vec_name]['ini_params']['use_state_knowledge_uncertainty'] = param
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_state_knowledge_uncertainty'] = False
        if 'return_belief_state_entropy' in conf['dst_sys'][dst_name]['ini_params']:
            param = conf['dst_sys'][dst_name]['ini_params']['return_belief_state_entropy']
            conf['vectorizer_sys'][vec_name]['ini_params']['use_state_total_uncertainty'] = param
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_state_total_uncertainty'] = False

    from convlab.nlu import NLU
    from convlab.dst import DST
    from convlab.policy import Policy
    from convlab.nlg import NLG

    modules = ['vectorizer_sys', 'nlu_sys', 'dst_sys', 'sys_nlg',
               'nlu_usr', 'dst_usr', 'policy_usr', 'usr_nlg']

    # Syncronise all seeds
    if 'seed' in conf['model']:
        for module in modules:
            module_name = [model for model in conf[module]]
            module_name = module_name[0] if module_name else None
            if conf[module] and module_name:
                if 'ini_params' in conf[module][module_name]:
                    if 'seed' in conf[module][module_name]['ini_params']:
                        conf[module][module_name]['ini_params']['seed'] = conf['model']['seed']

    # for each unit in modules above, create model save into conf
    for unit in modules:
        if conf[unit] == {}:
            conf[unit + '_activated'] = None
        else:
            for (model, infos) in conf[unit].items():
                cls_path = infos.get('class_path', '')
                cls = map_class(cls_path)
                conf[unit + '_class'] = cls
                conf[unit + '_activated'] = conf[unit + '_class'](**conf[unit][model]['ini_params'])
                print("Loaded " + model + " for " + unit)
    return conf


if __name__ == '__main__':
    get_goal_distribution()
