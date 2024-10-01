import os, json

from tqdm import tqdm

from convlab.base_models.t5.nlu.nlu import T5NLU

user_nlu = T5NLU(speaker='system', context_window_size=3, model_name_or_path='ConvLab/t5-small-nlu-all-multiwoz21-context3')

for sys_name in ['emoloop', 'emoloop_base', 'emoloop_express', 'emoloop_recognise']:
    for seed in ['0', '1', '2', '3', '4', '5']:
        log_f = f'{sys_name}/seed-{seed}.json'
        if not os.path.exists(log_f):
            print(f'{log_f} does not exist. Skip')
            continue
        sys_log = json.load(open(log_f, 'r'))
        print(f'processing {log_f}')

        save_f = f'{sys_name}/actions-{seed}.json'
        if os.path.exists(save_f):
            continue

        sys_actions = {}
        i = 0
        for item in tqdm(sys_log):
            dialog_id, _ = item['utt_idx'].split('_')
            dialog_id = dialog_id.replace('.json', '').lower()
            if dialog_id not in sys_actions:
                sys_actions[dialog_id] = []

            sys_utt = item['sys_utt']
            context = [t[1] for t in item['state']['history'][:-1]]
            act = user_nlu.predict(sys_utt, context=context)
            sys_actions[dialog_id].append(act)
        
        json.dump(sys_actions, open(save_f, 'w'), indent=4)

# dialog_score = evaluator.evaluate_dialog(goal, user_acts, system_acts, system_states)