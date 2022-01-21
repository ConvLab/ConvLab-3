import os
import sys
import json

from convlab2.util.custom_util import load_config_file


def map_class(cls_path: str):
    """
    Map to class via package text path
    :param cls_path: str, path with `convlab2` project directory as relative path, separator with `,`
                            E.g  `convlab2.nlu.svm.camrest.nlu.SVMNLU`
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
        if 'get_confidence_scores' in conf['dst_sys'][dst_name]['ini_params']:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_confidence_scores'] = conf['dst_sys'][dst_name]['ini_params']['get_confidence_scores']
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_confidence_scores'] = False
        if 'return_mutual_info' in conf['dst_sys'][dst_name]['ini_params']:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_mutual_info'] = conf['dst_sys'][dst_name]['ini_params']['return_mutual_info']
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_mutual_info'] = False
        if 'return_entropy' in conf['dst_sys'][dst_name]['ini_params']:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_entropy'] = conf['dst_sys'][dst_name]['ini_params']['return_entropy']
        else:
            conf['vectorizer_sys'][vec_name]['ini_params']['use_entropy'] = False

    from convlab2.nlu import NLU
    from convlab2.dst import DST
    from convlab2.policy import Policy
    from convlab2.nlg import NLG

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
                conf[unit + '_activated'] = conf[unit +
                                                 '_class'](**conf[unit][model]['ini_params'])
                print("Loaded " + model + " for " + unit)
    return conf


if __name__ == '__main__':
    # test
    args = [('model', 'seed', 'ThisIsATestSeed'),
            ('dst_sys', "setsumbt-mul", "ini_params", "get_confidence_scores", True)]
    path = "/Users/carel17/Projects/Convlab/convlab2/policy/ppo/setsumbt_config.json"
    conf = get_config(path, args)
    print(conf)
