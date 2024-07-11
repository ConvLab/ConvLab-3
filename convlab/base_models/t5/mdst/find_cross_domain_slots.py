import json
from copy import deepcopy
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from convlab.util import load_ontology
from convlab.base_models.t5.mdst.utils import get_state_update, SLOT_SIM_TH, VALUE_SIM_TH

def slot_sim(scores):
    if len(scores) == 2:
        return scores[1] > SLOT_SIM_TH
    else:
        return scores[2] > VALUE_SIM_TH

def find_same_value(state):
    # find slots that belong to different domains but have the same value (except true/false/yes/no)
    value2slot = {}
    for domain in state:
        for slot in state[domain]:
            value = state[domain][slot]
            if value in ['True', 'False', 'yes', 'no']:
                continue
            value2slot.setdefault(value, [])
            value2slot[value].append(f'{domain}-{slot}')
    return {value: value2slot[value] for value in value2slot if len(set([d_s.split('-')[0] for d_s in value2slot[value]])) > 1}

def get_same_value_slots(multi_domain_dials, domains):
    # get coref (same value for different slots) in multi-domain dialog
    # recall all but many have false positive (not 100% precise) like "stars" and "people"
    slot_coref = {}
    for dial in multi_domain_dials:
        slot_coref_dial = set()
        for turn in dial['turns']:
            if 'state' in turn:
                value2slot = find_same_value(turn['state'])
                for value, ds_list in value2slot.items():
                    ds_list = tuple(sorted(ds_list))
                    t = (value, ds_list)
                    if t not in slot_coref_dial:
                        slot_coref_dial.add(t)
        for value, ds_list in slot_coref_dial:
            slot_coref.setdefault(ds_list, [])
            slot_coref[ds_list].append(value)
    
    df = pd.DataFrame([],index=domains,columns=domains)
    for slots, values in slot_coref.items():
        cnt = len(values)
        if cnt < 10:
            continue
        for i in range(len(slots)):
            di, si = slots[i].split('-')
            for j in range(i+1, len(slots)):
                dj, sj = slots[j].split('-')
                if not isinstance(df.loc[di,dj], list):
                    df.loc[di,dj] = []
                pair = f'{si}-{sj}'
                exist_pairs = [x[0] for x in df.loc[di,dj]]
                if pair in exist_pairs:
                    idx = exist_pairs.index(pair)
                    df.loc[di,dj][idx] = (df.loc[di,dj][idx][0], df.loc[di,dj][idx][1]+cnt)
                else:
                    df.loc[di,dj].append((pair, cnt))
                df.loc[dj,di] = [(f"{item[0].split('-')[1]}-{item[0].split('-')[0]}", item[1]) for item in df.loc[di,dj]]
    return df

def predict_coref(dataset_name, single_domain_dials, classify_fn, args):
    random.seed(42)
    domain2slot2value = {}
    for dial in single_domain_dials:
        prev_state = {}
        for turn in dial['turns']:
            if 'state' in turn:
                state_update = get_state_update(prev_state, turn['state'])
                for domain in state_update:
                    domain2slot2value.setdefault(domain, {})
                    for slot, value in state_update[domain].items():
                        domain2slot2value[domain].setdefault(slot, [])
                        domain2slot2value[domain][slot].append(value)
                prev_state = turn['state']
    num_sample_value = 10
    # randomly sample 10 values for each slot
    for domain in domain2slot2value:
        for slot, value_set in domain2slot2value[domain].items():
            domain2slot2value[domain][slot] = random.sample(value_set, min(num_sample_value, len(value_set)))

    domains = sorted(list(domain2slot2value.keys()))
            
    model = SentenceTransformer(args.embed_model)
    ontology = load_ontology(dataset_name)
    
    # embed slot name, description, and values
    domain2slot2embed = {}
    for domain in domains:
        domain2slot2embed[domain] = {}
        for slot in domain2slot2value[domain]:
            name_embed = model.encode(' '.join(slot.split('_')))
            desc = ontology['domains'][domain]['slots'][slot]['description']
            desc_embed = model.encode(desc)
            if any([value in ['True', 'False', 'yes', 'no'] for value in domain2slot2value[domain][slot]]):
                # binary slot without value embedding
                domain2slot2embed[domain][slot] = (name_embed, desc_embed)
            else:
                values = domain2slot2value[domain][slot]
                value_embed = np.mean(model.encode(values),axis=0)
                domain2slot2embed[domain][slot] = (name_embed, desc_embed, value_embed)

    df = pd.DataFrame([],index=domains,columns=domains)
    all_scores_df = pd.DataFrame([],columns=['domain-slot1', 'domain-slot2', 'score', 'binary'])
    for i in range(len(domains)):
        embed_i = domain2slot2embed[domains[i]]
        for j in range(i+1, len(domains)):
            embed_j = domain2slot2embed[domains[j]]
            sim_mat = {}
            for slot_i in embed_i:
                for slot_j in embed_j:
                    sim_scores = []
                    if len(embed_i[slot_i]) != len(embed_j[slot_j]):
                        continue
                    for k in range(len(embed_i[slot_i])):
                        sim_score = util.cos_sim(embed_i[slot_i][k], embed_j[slot_j][k]).item()
                        sim_scores.append(sim_score)
                    all_scores_df.loc[len(all_scores_df.index)] = [f'{domains[i]}-{slot_i}', f'{domains[j]}-{slot_j}', sim_scores[-1], len(sim_scores) == 2]
                    if classify_fn(sim_scores):
                        sim_mat[f'{slot_i}-{slot_j}'] = sim_scores
            if len(sim_mat) > 0:
                df.iloc[i,j] = list(sim_mat.items())
                df.iloc[j,i] = [(f"{item[0].split('-')[1]}-{item[0].split('-')[0]}", item[1]) for item in df.iloc[i,j]]
    return df, all_scores_df

def eval_coref(predict_df, same_value_df):
    TP, FP, FN = 0, 0, 0
    domains = list(predict_df.columns)
    df = pd.DataFrame([],index=domains,columns=domains)
    for i in range(len(domains)):
        for j in range(i+1,len(domains)):
            if isinstance(same_value_df.iloc[i,j], list):
                same_value_slots = set([x[0] for x in same_value_df.iloc[i,j]])
            else:
                continue
            if isinstance(predict_df.iloc[i,j], list):
                predict_slots = set([x[0] for x in predict_df.iloc[i,j]])
            else:
                predict_slots = set()
            tp = len(predict_slots & same_value_slots)
            fp = len(predict_slots - same_value_slots)
            fn = len(same_value_slots - predict_slots)
            precision = 1.0 * tp / (tp + fp) if tp + fp else 0.
            recall = 1.0 * tp / (tp + fn) if tp + fn else 0.
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
            df.iloc[i,j] = (precision, recall, f1)
            df.iloc[j,i] = df.iloc[i,j]
            TP += tp
            FP += fp
            FN += fn
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    print(f"TP={TP}, FP={FP}, FN={FN}, precision={precision}, recall={recall}, f1={f1}")
    return df

def find_cross_domain_slots(dataset_name, data_dir, args):
    single_domain_dials = json.load(open(os.path.join(data_dir, 'single_domain.json')))
    multi_domain_dials = json.load(open(os.path.join(data_dir, 'multi_domain.json')))
    predict_df, all_scores_df = predict_coref(dataset_name, single_domain_dials, slot_sim, args)
    predict_df.to_csv(os.path.join(data_dir, 'predict_coref.csv'))
    all_scores_df.to_csv(os.path.join(data_dir, 'all_scores.csv'))
    same_value_df = get_same_value_slots(multi_domain_dials, list(predict_df.columns))
    same_value_df.to_csv(os.path.join(data_dir, 'same_value_coref.csv'))
    eval_df = eval_coref(predict_df, same_value_df)
    eval_df.to_csv(os.path.join(data_dir, 'eval_coref.csv'))
    


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--group_idx', '-g', type=int, nargs='*', default=None, help='group index (for sgd)')
    parser.add_argument('--embed_model', '-m', metavar='embedding model', default='/zhangpai23/zhuqi/pre-trained-models/all-mpnet-base-v2', help='path of the embedding model of Sentence Transformers')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        if args.group_idx is not None:
            for group_idx in args.group_idx:
                data_dir = os.path.join('data', f'{dataset_name}/group{group_idx}')
                find_cross_domain_slots(dataset_name, data_dir, args)
        else:
            data_dir = os.path.join('data', dataset_name)
            find_cross_domain_slots(dataset_name, data_dir, args)
