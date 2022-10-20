import copy
import json
import os
import re
from collections import Counter
from pprint import pprint
from shutil import copy2, rmtree
from zipfile import ZIP_DEFLATED, ZipFile

from tqdm import tqdm

ontology = {
    "domains": {
        "景点": {
            "description": "查找景点",
            "slots": {
                "名称": {
                    "description": "景点名称",
                    "is_categorical": False,
                    "possible_values": []
                },
                "门票": {
                    "description": "景点门票价格",
                    "is_categorical": False,
                    "possible_values": []
                },
                "游玩时间": {
                    "description": "景点游玩时间",
                    "is_categorical": False,
                    "possible_values": []
                },
                "评分": {
                    "description": "景点评分",
                    "is_categorical": False,
                    "possible_values": []
                },
                "地址": {
                    "description": "景点地址",
                    "is_categorical": False,
                    "possible_values": []
                },
                "电话": {
                    "description": "景点电话",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边景点": {
                    "description": "景点周边景点",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边餐馆": {
                    "description": "景点周边餐馆",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边酒店": {
                    "description": "景点周边酒店",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "餐馆": {
            "description": "查找餐馆",
            "slots": {
                "名称": {
                    "description": "餐馆名称",
                    "is_categorical": False,
                    "possible_values": []
                },
                "推荐菜": {
                    "description": "餐馆推荐菜",
                    "is_categorical": False,
                    "possible_values": []
                },
                "人均消费": {
                    "description": "餐馆人均消费",
                    "is_categorical": False,
                    "possible_values": []
                },
                "评分": {
                    "description": "餐馆评分",
                    "is_categorical": False,
                    "possible_values": []
                },
                "地址": {
                    "description": "餐馆地址",
                    "is_categorical": False,
                    "possible_values": []
                },
                "电话": {
                    "description": "餐馆电话",
                    "is_categorical": False,
                    "possible_values": []
                },
                "营业时间": {
                    "description": "餐馆营业时间",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边景点": {
                    "description": "餐馆周边景点",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边餐馆": {
                    "description": "餐馆周边餐馆",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边酒店": {
                    "description": "餐馆周边酒店",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "酒店": {
            "description": "查找酒店",
            "slots": {
                "名称": {
                    "description": "酒店名称",
                    "is_categorical": False,
                    "possible_values": []
                },
                "酒店类型": {
                    "description": "酒店类型",
                    "is_categorical": True,
                    "possible_values": [
                        '高档型', '豪华型', '经济型', '舒适型'
                    ]
                },
                "酒店设施": {
                    "description": "酒店设施",
                    "is_categorical": False,
                    "possible_values": []
                },
                "价格": {
                    "description": "酒店价格",
                    "is_categorical": False,
                    "possible_values": []
                },
                "评分": {
                    "description": "酒店评分",
                    "is_categorical": False,
                    "possible_values": []
                },
                "地址": {
                    "description": "酒店地址",
                    "is_categorical": False,
                    "possible_values": []
                },
                "电话": {
                    "description": "酒店电话",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边景点": {
                    "description": "酒店周边景点",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边餐馆": {
                    "description": "酒店周边餐馆",
                    "is_categorical": False,
                    "possible_values": []
                },
                "周边酒店": {
                    "description": "酒店周边酒店",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "地铁": {
            "description": "乘坐地铁从某地到某地",
            "slots": {
                "出发地": {
                    "description": "地铁出发地",
                    "is_categorical": False,
                    "possible_values": []
                },
                "目的地": {
                    "description": "地铁目的地",
                    "is_categorical": False,
                    "possible_values": []
                },
                "出发地附近地铁站": {
                    "description": "出发地附近地铁站",
                    "is_categorical": False,
                    "possible_values": []
                },
                "目的地附近地铁站": {
                    "description": "目的地附近地铁站",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "出租": {
            "description": "乘坐出租车从某地到某地",
            "slots": {
                "出发地": {
                    "description": "出租出发地",
                    "is_categorical": False,
                    "possible_values": []
                },
                "目的地": {
                    "description": "出租目的地",
                    "is_categorical": False,
                    "possible_values": []
                },
                "车型": {
                    "description": "出租车车型",
                    "is_categorical": True,
                    "possible_values": [
                        "#CX"
                    ]
                },
                "车牌": {
                    "description": "出租车车牌",
                    "is_categorical": True,
                    "possible_values": [
                        "#CP"
                    ]
                }
            }
        },
        "General": {
            "description": "通用领域",
            "slots": {}
        }
    },
    "intents": {
        "Inform": {
            "description": "告知相关属性"
        },
        "Request": {
            "description": "询问相关属性"
        },
        "Recommend": {
            "description": "推荐搜索结果"
        },
        "Select": {
            "description": "在附近搜索"
        },
        "NoOffer": {
            "description": "未找到符合用户要求的结果"
        },
        "bye": {
            "description": "再见"
        },
        "thank": {
            "description": "感谢"
        },
        "welcome": {
            "description": "不客气"
        },
        "greet": {
            "description": "打招呼"
        },
    },
    "state": {
        "景点": {
            "名称": "",
            "门票": "",
            "游玩时间": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
        },
        "餐馆": {
            "名称": "",
            "推荐菜": "",
            "人均消费": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
        },
        "酒店": {
            "名称": "",
            "酒店类型": "",
            "酒店设施": "",
            "价格": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
        },
        "地铁": {
            "出发地": "",
            "目的地": "",
        },
        "出租": {
            "出发地": "",
            "目的地": "",
        }
    },
    "dialogue_acts": {
        "categorical": {},
        "non-categorical": {},
        "binary": {}
    }
}

cnt_domain_slot = Counter()

def convert_da(da_list, utt):
    '''
    convert dialogue acts to required format
    :param da_dict: list of (intent, domain, slot, value)
    :param utt: user or system utt
    '''
    global ontology, cnt_domain_slot

    converted_da = {
        'categorical': [],
        'non-categorical': [],
        'binary': []
    }

    for intent, domain, slot, value in da_list:
        # if intent in ['Inform', 'Recommend']:
        if intent == 'NoOffer':
            assert slot == 'none' and value == 'none'
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': ''
            })
        elif intent == 'General':
            # intent=General, domain=thank/bye/greet/welcome
            assert slot == 'none' and value == 'none'
            converted_da['binary'].append({
                'intent': domain,
                'domain': intent,
                'slot': ''
            })
        elif intent == 'Request':
            assert value == '' and slot != 'none'
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot
            })
        elif '酒店设施' in slot:
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': f"{slot}-{value}"
            })
        elif intent == 'Select':
            assert slot == '源领域'
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': f"{slot}-{value}"
            })
        elif slot in ['酒店类型', '车型', '车牌']:
            assert intent in ['Inform', 'Recommend']
            assert slot != 'none' and value != 'none'
            converted_da['categorical'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot,
                'value': value
            })
        else:
            assert intent in ['Inform', 'Recommend']
            assert slot != 'none' and value != 'none'
            matches = utt.count(value)
            if matches == 1:
                start = utt.index(value)
                end = start + len(value)
                
                converted_da['non-categorical'].append({
                    'intent': intent,
                    'domain': domain,
                    'slot': slot,
                    'value': value,
                    'start': start,
                    'end': end
                })
                cnt_domain_slot['have span'] += 1
            else:
                # can not find span
                converted_da['non-categorical'].append({
                    'intent': intent,
                    'domain': domain,
                    'slot': slot,
                    'value': value
                })
                cnt_domain_slot['no span'] += 1
            # cnt_domain_slot.setdefault(f'{domain}-{slot}', set())
            # cnt_domain_slot[f'{domain}-{slot}'].add(value)
        
    return converted_da

def transform_user_state(user_state):
    goal = []
    for subgoal in user_state:
        gid, domain, slot, value, mentioned = subgoal
        if len(value) != 0:
            t = 'inform'
        else:
            t = 'request'
        if len(goal) < gid:
            goal.append({domain: {'inform': {}, 'request': {}}})
        goal[gid-1][domain][t][slot] = [value, 'mentioned' if mentioned else 'not mentioned']
    return goal


def preprocess():
    original_data_dir = '../../crosswoz'
    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)
    for filename in os.listdir(os.path.join(original_data_dir,'database')):
        copy2(f'{original_data_dir}/database/{filename}', new_data_dir)

    global ontology

    dataset = 'crosswoz'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split: [] for split in splits}
    for split in ['train', 'val', 'test']:
        data = json.load(ZipFile(os.path.join(original_data_dir, f'{split}.json.zip'), 'r').open(f'{split}.json'))
        if split == 'val':
            split = 'validation'
    
        for ori_dialog_id, ori_dialog in data.items():
            if ori_dialog_id in ['10550', '11724']:
                # skip error dialog
                continue
            dialogue_id = f'{dataset}-{split}-{len(dialogues_by_split[split])}'

            # get user goal and involved domains
            goal = {'inform': {}, 'request': {}}
            goal["description"] = '\n'.join(ori_dialog["task description"])
            cur_domains = [x[1] for i, x in enumerate(ori_dialog['goal']) if i == 0 or ori_dialog['goal'][i-1][1] != x[1]]

            dialogue = {
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': dialogue_id,
                'original_id': ori_dialog_id,
                'domains': cur_domains,
                'goal': goal,
                'user_state_init': transform_user_state(ori_dialog['goal']),
                'type': ori_dialog['type'],
                'turns': [],
                'user_state_final': transform_user_state(ori_dialog['final_goal'])
            }

            for turn_id, turn in enumerate(ori_dialog['messages']):
                if ori_dialog_id == '2660' and turn_id in [8,9]:
                    # skip error turns
                    continue
                elif ori_dialog_id == '7467' and turn_id in [14,15]:
                    # skip error turns
                    continue
                elif ori_dialog_id == '11726' and turn_id in [4,5]:
                    # skip error turns
                    continue
                speaker = 'user' if turn['role'] == 'usr' else 'system'
                utt = turn['content']

                das = turn['dialog_act']

                dialogue_acts = convert_da(das, utt)

                dialogue['turns'].append({
                    'speaker': speaker,
                    'utterance': utt,
                    'utt_idx': len(dialogue['turns']),
                    'dialogue_acts': dialogue_acts,
                })

                # add to dialogue_acts dictionary in the ontology
                for da_type in dialogue_acts:
                    das = dialogue_acts[da_type]
                    for da in das:
                        ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                        ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])][speaker] = True

                if speaker == 'user':
                    dialogue['turns'][-1]['user_state'] = transform_user_state(turn['user_state'])
                else:
                    # add state to last user turn
                    belief_state = turn['sys_state_init']
                    for domain in belief_state:
                        belief_state[domain].pop('selectedResults')
                    dialogue['turns'][-2]['state'] = belief_state
                    db_query = turn['sys_state']
                    db_results = {}
                    for domain in list(db_query.keys()):
                        db_res = db_query[domain].pop('selectedResults')
                        if len(db_res) > 0:
                            db_results[domain] = [{'名称': x} for x in db_res]
                        else:
                            db_query.pop(domain)
                    dialogue['turns'][-1]['db_query'] = db_query
                    dialogue['turns'][-1]['db_results'] = db_results
            dialogues_by_split[split].append(dialogue)
    pprint(cnt_domain_slot.most_common())
    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str(
            {'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent': da[0],
             'domain': da[1], 'slot': da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(new_data_dir)
    return dialogues, ontology


def fix_entity_booked_info(entity_booked_dict, booked):
    for domain in entity_booked_dict:
        if not entity_booked_dict[domain] and booked[domain]:
            entity_booked_dict[domain] = True
            booked[domain] = []
    return entity_booked_dict, booked


if __name__ == '__main__':
    preprocess()