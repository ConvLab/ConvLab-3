import os
import sys
import json
import logging


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
    for sec in ['model', 'nlu_sys', 'dst_sys', 'sys_nlg', 'nlu_usr', 'policy_usr', 'usr_nlg']:
        assert sec in conf.keys(), 'Missing \'%s\' section in config file \'%s\'' % (sec, filepath)

    return conf


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
    # load settings
    if filepath == 'default':
        filepath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'envs_config.json'))

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

    # Auto load uncertinty settings from policy based on the tracker used
    dst_name = [model for model in conf['dst_sys']]
    dst_name = dst_name[0] if dst_name else None
    if dst_name and 'setsumbt' in dst_name.lower():
        if 'get_confidence_scores' in conf['dst_sys'][dst_name]['ini_params']:
            conf['model']['use_confidence_scores'] = conf['dst_sys'][dst_name]['ini_params']['get_confidence_scores']
        else:
            conf['model']['use_confidence_scores'] = False
        if 'return_mutual_info' in conf['dst_sys'][dst_name]['ini_params']:
            conf['model']['use_state_mutual_info'] = conf['dst_sys'][dst_name]['ini_params']['return_mutual_info']
        else:
            conf['model']['use_state_mutual_info'] = False
        if 'return_entropy' in conf['dst_sys'][dst_name]['ini_params']:
            conf['model']['use_state_entropy'] = conf['dst_sys'][dst_name]['ini_params']['return_entropy']
        else:
            conf['model']['use_state_entropy'] = False

    # Auto load NLG seed from model
    usr_nlg_name = [model for model in conf['usr_nlg']]
    usr_nlg_name = usr_nlg_name[0] if usr_nlg_name else None
    if usr_nlg_name and 'template' in usr_nlg_name.lower():
        conf['usr_nlg'][usr_nlg_name]['ini_params']['seed'] = conf['model']['seed']

    from convlab.nlu import NLU
    from convlab.dst import DST
    from convlab.policy import Policy
    from convlab.nlg import NLG

    modules = ['nlu_sys', 'dst_sys', 'sys_nlg',
               'nlu_usr', 'dst_usr', 'policy_usr', 'usr_nlg']

    # for each unit in modules above, create model save into conf
    for unit in modules:
        if conf[unit] == {}:
            #logging.info("Warning: No " + unit + "is used")
            conf[unit + '_activated'] = None
        else:
            for (model, infos) in conf[unit].items():
                cls_path = infos.get('class_path', '')
                cls = map_class(cls_path)
                conf[unit + '_class'] = cls
                conf[unit + '_activated'] = conf[unit +
                                                 '_class'](**conf[unit][model]['ini_params'])
                logging.info("Loaded " + model + " for " + unit)
    return conf


if __name__ == '__main__':
    # test
    args = [('model', 'seed', 'ThisIsATestSeed'),
            ('dst_sys', "setsumbt-mul", "ini_params", "get_confidence_scores", True)]
    path = "/Users/carel17/Projects/Convlab/convlab/policy/ppo/setsumbt_config.json"
    conf = get_config(path, args)
    print(conf)
