"""
Evaluate NLU models on specified dataset
Metric: dataset level Precision/Recall/F1
Usage: python evaluate.py [MultiWOZ] [SCLSTM|TemplateNLG] [usr|sys]
"""

import json
import os
import random
import sys
import itertools
import zipfile
import numpy
from numpy.lib.shape_base import _put_along_axis_dispatcher
from numpy.lib.twodim_base import triu_indices_from
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pprint import pprint
from tqdm import tqdm


def slot_error(dialog_acts, utts):
    halucination = []
    halucinate = 0
    missing = 0
    total = 0

    for acts,utt in zip(dialog_acts, utts):
        for act in acts:
            tmp_act = [x.lower() for x in act]
            tmp_utt = utt.lower()
            i, d, s, v = tmp_act
            if i == 'inform':
                total = total + 1
                if not (v in tmp_utt):
                    missing = missing + 1
    return missing, total

def fine_SER(dialog_acts, utts):
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'template', 'multiwoz', 'label_maps.json')
    with open(path, 'r') as mapping_file:
        mappings = json.load(mapping_file)
        mapping_file.close()

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(path, 'data', 'multiwoz', 'ontology_nlg_eval.json')
    with open(path, 'r') as entity_file:
        possible_entity = json.load(entity_file)
        entity_file.close()

    entity_list = []

    for key in possible_entity.keys():
        entity_list = entity_list + possible_entity[key]
    
    hallucinate = 0
    missing = 0
    total = 0

    unk_token_count = 0
    missing_dialogs = []
    hallucination_dialogs = []

    slot_span = []
    domain_span = []

    for acts,utt in zip(dialog_acts, utts):
        hallucination_flag = False        
        tmp_utt = utt.lower()
        origin_utt = utt.lower()
        legal_act_flag = False

        for act in acts:
            missing_fact = None
            missing_flag = False
            tmp_act = [x.lower() for x in act]
            i, d, s, v = tmp_act

            if not(d in domain_span):
                domain_span.append(d)
            if not(s in slot_span):
                slot_span.append(s)
            #intializing all possible span keyword

            if i in ['inform', 'recommend', 'offerbook', 'offerbooked','book','select']:
                legal_act_flag = True
                total = total + 1
                if not (v in origin_utt) and v!='none':
                    exist_flag = False
                    try:
                        synoyms = mappings[v]
                        for item in synoyms:
                            if item in origin_utt:
                                exist_flag = True
                                tmp_utt = tmp_utt.replace(item,'')
                                tmp_utt = tmp_utt.replace(s,'')
                                #remove span for hallucination detection
                    except:
                        pass
                    if i in ['offerbook', 'offerbooked'] and v =='none':
                        if 'book' in origin_utt:
                            exist_flag = True
                            tmp_utt = tmp_utt.replace('book','')
                    if i in ['inform','recommend'] and v=='none':
                        if d in origin_utt:
                            exist_flag = True
                            tmp_utt = tmp_utt.replace(d, '')
                    if exist_flag == False:
                        missing_flag = True
                        missing_fact = v
                else:
                    tmp_utt = tmp_utt.replace(v,'')
                    tmp_utt = tmp_utt.replace(s,'')

                if s in origin_utt:
                    missing_flag = False
                if s =='booking' and ('book' in origin_utt or 'reserv' in origin_utt):
                    missing_flag = False

            elif i == 'request':
                legal_act_flag = True
                total = total + 1
                if s=='depart' or s=='dest' or s=='area':
                    if not ('where' in origin_utt):
                        if s in origin_utt:
                            tmp_utt = tmp_utt.replace(s,'')
                        else:
                            missing_flag = True
                            missing_fact = s
                elif s=='leave' or s=='arrive':
                    if (not 'when' in origin_utt):
                        if not ('what' in origin_utt and 'time' in origin_utt):
                            missing_flag = True
                            missing_fact = s
                    else:
                        tmp_utt.replace('time', '')
                else:
                    tmp_utt = tmp_utt.replace(s,'')
                    tmp_utt = tmp_utt.replace(d,'')

                if s in origin_utt:
                        missing_flag = False
                if s =='booking' and ('book' in origin_utt or 'reserv' in origin_utt):
                    missing_flag = False    

            try:
                tmp_utt = tmp_utt.replace(d,'')
                tmp_utt = tmp_utt.replace(s,'')
                if 'arrive' in s or 'leave' in s:
                    tmp_utt = tmp_utt.replace('time', '')
            except:
                pass

            if missing_flag == True:
                missing = missing + 1
                missing_dialogs.append(missing_fact)
                missing_dialogs.append(acts)
                missing_dialogs.append(utt)

        for keyword in slot_span + entity_list:
            if keyword in tmp_utt and len(keyword) >= 4 and legal_act_flag == True:
                hallucination_flag = True
                hallucinate = hallucinate + 1
                hallucination_dialogs.append(keyword)
                hallucination_dialogs.append(acts)
                hallucination_dialogs.append(tmp_utt)
                hallucination_dialogs.append(utt)
                break


    return missing, hallucinate, total, hallucination_dialogs, missing_dialogs


def get_bleu4(dialog_acts, golden_utts, gen_utts):
    das2utts = {}
    for das, utt, gen in zip(dialog_acts, golden_utts, gen_utts):
        utt = utt.lower()
        gen = gen.lower()
        for da in das:
            act, domain, s, v = da
            if act == 'Request' or domain == 'general':
                continue
            else:
                if s == 'Internet' or s == 'Parking' or s == 'none' or v == 'none':
                    continue
                else:
                    v = v.lower()
                    if (' ' + v in utt) or (v + ' ' in utt):
                        utt = utt.replace(v, '{}-{}'.format(act + '-' + domain, s), 1)
                    if (' ' + v in gen) or (v + ' ' in gen):
                        gen = gen.replace(v, '{}-{}'.format(act + '-' + domain, s), 1)
        hash_key = ''
        for da in sorted(das, key=lambda x: x[0] + x[1] + x[2]):
            hash_key += '-'.join(da[:-1]) + ';'
        das2utts.setdefault(hash_key, {'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append(gen)
    # pprint(das2utts)
    refs, gens = [], []
    for das in das2utts.keys():
        for gen in das2utts[das]['gens']:
            refs.append([s.split() for s in das2utts[das]['refs']])
            gens.append(gen.split())
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return bleu


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if len(sys.argv) < 4:
        print("usage:")
        print("\t python evaluate.py dataset model role")
        print("\t dataset=MultiWOZ, CrossWOZ, or Camrest")
        print("\t model=SCLSTM, SCLSTM_NoUNK, SCGPT or TemplateNLG")
        print("\t role=usr/sys")
        print("\t [Optional] model_file")
        sys.exit()
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    role = sys.argv[3]
    model_file = sys.argv[4] if len(sys.argv) >= 5 else None
    if dataset_name == 'MultiWOZ':
        if model_name == 'SCLSTM':
            from convlab.nlg.sclstm.multiwoz import SCLSTM
            if role == 'usr':
                model = SCLSTM(is_user=True, use_cuda=True, unk_suppress=False)
            elif role == 'sys':
                model = SCLSTM(is_user=False, use_cuda=True, unk_suppress=False)
        elif model_name == 'SCLSTM_NoUNK':
            from convlab.nlg.sclstm.multiwoz import SCLSTM
            if role == 'usr':
                model = SCLSTM(is_user=True, use_cuda=True, unk_suppress=True)
            elif role == 'sys':
                model = SCLSTM(is_user=False, use_cuda=True, unk_suppress=True)
        elif model_name == 'TemplateNLG':
            from convlab.nlg.template.multiwoz import TemplateNLG
            if role == 'usr':
                model = TemplateNLG(is_user=True)
            elif role == 'sys':
                model = TemplateNLG(is_user=False)
        elif model_name == 'SCGPT':
            from convlab.nlg.scgpt.multiwoz import SCGPT
            if model_file is not None:
                print(f"load model at {model_file}")
            if role == 'usr':
                model = SCGPT(model_file, is_user=True)
            elif role == 'sys':
                model  = SCGPT(model_file, is_user=False)
        else:
            raise Exception("Available models: SCLSTM, SCGPT, TEMPLATE")

        from convlab.util.dataloader.module_dataloader import SingleTurnNLGDataloader
        from convlab.util.dataloader.dataset_dataloader import MultiWOZDataloader
        dataloader = SingleTurnNLGDataloader(dataset_dataloader=MultiWOZDataloader())
        data = dataloader.load_data(data_key='all', role=role, session_id=True)['test']

        dialog_acts = []
        golden_utts = []
        gen_utts = []
        gen_slots = []

        sen_num = 0

        # sys.stdout = open(sys.argv[2] + '-' + sys.argv[3] + '-' + 'evaluate_logs_neo.txt','w')
        assert 'utterance' in data and 'dialog_act' in data and 'session_id' in data
        assert len(data['utterance']) == len(data['dialog_act']) == len(data['session_id'])

        # Turns during the same session should be contiguous, so we can call init_session at the first turn of a new session.
        # This is necessary for SCGPT, but unnecessary for SCLSTM and TemplateNLG.
        is_first_turn = []
        for _, iterator in itertools.groupby(data['session_id']):
            is_first_turn.append(True)
            next(iterator)
            is_first_turn.extend(False for _ in iterator)
        for i in tqdm(range(len(data['utterance']))):
            if is_first_turn[i]:
                model.init_session()
            dialog_acts.append(data['dialog_act'][i])
            golden_utts.append(data['utterance'][i])
            gen_utts.append(model.generate(data['dialog_act'][i]))
        #     print(dialog_acts[-1])
        #     print(golden_utts[-1])
        #     print(gen_utts[-1])

        print("Calculate SER for golden responses")
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(dialog_acts, golden_utts)
        print("Golden response Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(missing, total, hallucinate, missing/total))
        
        print("Calculate SER")
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(dialog_acts, gen_utts)
        # with open('{}-{}-genutt_neo.txt'.format(sys.argv[2], sys.argv[3]), mode='wt', encoding='utf-8') as gen_diag:
        #     for x in gen_utts:
        #         gen_diag.writelines(str(x)+'\n')


        # with open('{}-{}-hallucinate_neo.txt'.format(sys.argv[2], sys.argv[3]), mode='wt', encoding='utf-8') as hal_diag:
        #     for x in hallucination_dialogs:
        #         hal_diag.writelines(str(x)+'\n')
        
        # with open('{}-{}-missing_neo.txt'.format(sys.argv[2], sys.argv[3]), mode='wt', encoding='utf-8') as miss_diag:
        #     for x in missing_dialogs:
        #         miss_diag.writelines(str(x)+'\n')
        print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(sys.argv[2], missing, total, hallucinate, missing/total))
        print("Calculate bleu-4")
        bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts)
        print("BLEU-4: %.4f" % bleu4)
        print('Model on {} sentences role={}'.format(len(data['utterance']), role))
        # sys.stdout.close()

    else:
        raise Exception("currently supported dataset: MultiWOZ")
