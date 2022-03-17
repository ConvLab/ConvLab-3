import re
import torch


def is_slot_da(da_type):
    return da_type == 'non-categorical'


def calculateF1(predict_golden):
    # F1 of all three types of dialogue acts
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        for da_type in ['non-categorical', 'categorical', 'binary']:
            if da_type not in item['predict']:
                assert da_type not in item['golden']
                continue
            if da_type == 'binary':
                predicts = [(x['intent'], x['domain'], x['slot']) for x in item['predict'][da_type]]
                labels = [(x['intent'], x['domain'], x['slot']) for x in item['golden'][da_type]]
            else:
                predicts = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in item['predict'][da_type]]
                labels = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in item['golden'][da_type]]
            
            for ele in predicts:
                if ele in labels:
                    TP += 1
                else:
                    FP += 1
            for ele in labels:
                if ele not in predicts:
                    FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


def tag2triples(word_seq, tag_seq):
    word_seq = word_seq[:len(tag_seq)]
    assert len(word_seq)==len(tag_seq)
    triples = []
    i = 0
    while i < len(tag_seq):
        tag = eval(tag_seq[i])
        if tag[-1] == 'B':
            intent, domain, slot = tag[0], tag[1], tag[2]
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                next_tag = eval(tag_seq[j])
                if next_tag[-1] == 'I' and next_tag[:-1] == tag[:-1]:
                    value += ' ' + word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            triples.append([intent, domain, slot, value])
        i += 1
    return triples


def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    # new2ori = {(new_idx:old_idx),...} (after removing [CLS] and [SEP]
    max_seq_len = tag_logits.size(0)
    dialogue_acts = {
        "categorical": [],
        "non-categorical": [],
        "binary": []
    }
    # for categorical & binary dialogue acts
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent = eval(dataloader.id2intent[j])
            if len(intent) == 3:
                dialogue_acts['binary'].append({
                    'intent': intent[0],
                    'domain': intent[1],
                    'slot': intent[2]
                })
            else:
                assert len(intent) == 4
                dialogue_acts['categorical'].append({
                    'intent': intent[0],
                    'domain': intent[1],
                    'slot': intent[2],
                    'value': intent[3]
                })
    # for non-categorical dialogues acts
    tags = []
    for j in range(1, max_seq_len-1):
        if tag_mask_tensor[j] == 1:
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    recover_tags = []
    for i, tag in enumerate(tags):
        if new2ori[i] >= len(recover_tags):
            recover_tags.append(tag)
    ori_word_seq = ori_word_seq[:len(recover_tags)]
    tag_intent = tag2triples(ori_word_seq, recover_tags)
    for intent in tag_intent:
        dialogue_acts['non-categorical'].append({
            'intent': intent[0],
            'domain': intent[1],
            'slot': intent[2],
            'value': intent[3]
        })
    return dialogue_acts
