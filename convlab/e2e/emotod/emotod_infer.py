# -*- coding: utf-8 -*-
import json
from copy import deepcopy

from emotod import EMOTODAgent
from utils import lexcalise

with open('corpus_eval/emo_prev.json', 'r') as f:
    dataset = json.load(f)

dataset_out = deepcopy(dataset)

s = EMOTODAgent(model_file='/home/fengs/projects/pretrained_models/gpt2_prev_emo')

model = s.model
tokenizer = s.tokenizer

for split in ['train', 'valid', 'test']:
    dialogs = dataset[split]

    for dial_id in dialogs:
        turns = dialogs[dial_id]
        for i in range(len(turns)):
            input = turns[i]['input_context']
            encoding = s.tokenizer(input, return_tensors="pt", padding=True).to(s.device)
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, 
                max_new_tokens=500, 
                eos_token_id=tokenizer.convert_tokens_to_ids([s.eos_token])[0],
                no_repeat_ngram_size=10,
                )
            
            full_generation = tokenizer.decode(outputs[0], skip_special_tokens=False)

            lex_resp = lexcalise(full_generation, s.database)
            dataset_out[split][dial_id][i]['output_delex'] = full_generation
            dataset_out[split][dial_id][i]['output_lex'] = lex_resp

with open('corpus_eval/multiwoz21_emotod_resp.json', 'w') as f:
    json.dump(dataset_out, f, indent=4)
