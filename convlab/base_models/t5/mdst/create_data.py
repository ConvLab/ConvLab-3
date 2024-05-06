"""generate data for model training"""
import os
import json
import pandas as pd
from tqdm import tqdm
import random
import shutil
from tabulate import tabulate
from convlab.util import load_ontology
from convlab.base_models.t5.mdst.data_processor import DataProcessor
from convlab.base_models.t5.mdst.utils import dials2qadst_samples, dials2qadst_cross_domain_samples, dials2domain_cls_samples, label_str


def create_single_domain_qadst_data(args):
    random.seed(42)
    src_data_dir = args.src_data_dir
    write_data_dir = args.write_data_dir
    ontology = load_ontology(args.dataset_name)
    full_state = json.load(open(os.path.join(src_data_dir, 'full_state.json')))
    for filename in ['train_single_domain.json', 'validation_single_domain.json', 'test_single_domain.json']:
        data = json.load(open(os.path.join(src_data_dir, filename)))

        # prepare single domain samples for qa-dst model
        output_filename = os.path.join(write_data_dir, filename.replace('_single_domain', ''))
        with open(output_filename, "w", encoding='utf-8') as f:
            samples = dials2qadst_samples(data, ontology, full_state)
            f.writelines(samples)

        if filename in ['train_single_domain.json', 'validation_single_domain.json']:
            output_filename = os.path.join(write_data_dir, filename.replace('single_domain', 'single_domain_qa'))
            with open(output_filename, "w", encoding='utf-8') as f:
                samples = dials2qadst_cross_domain_samples(data, ontology, full_state)
                f.writelines(samples)


def create_aug_data(args):
    assert args.data_aug is not None and args.data_aug_times is not None
    src_data_dir = args.src_data_dir
    read_data_dir = args.read_data_dir
    write_data_dir = args.write_data_dir
    table = []
    ontology = load_ontology(args.dataset_name)
    full_state = json.load(open(os.path.join(src_data_dir, 'full_state.json')))
    data_processor = DataProcessor(args.model_type, args.data_aug, args.context_window_size, ontology, full_state)
    for data_split in ['train', 'validation']:
        single_domain_dials = json.load(open(os.path.join(src_data_dir, f'{data_split}_single_domain.json')))

        # augment multi-domain dialogs from single domain dialogs
        if args.data_aug == DataProcessor.AUG_TYPE_NONE:
            # no data augmentation, only original single domain dialogs
            aug_dials = []
        else:
            kwargs = {}
            if args.data_aug == DataProcessor.AUG_TYPE_REPLACE_TRUE:
                multi_domain_dials = json.load(open(os.path.join(src_data_dir, f'{data_split}_multi_domain.json')))
            elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2, DataProcessor.AUG_TYPE_CONCATN]:
                multi_domain_dials = []
            elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2REL, DataProcessor.AUG_TYPE_CONCATNREL]:
                single_domain_dials = json.load(open(os.path.join(read_data_dir, f'{data_split}_single_domain.json')))
                slot_pairs = ['-'.join(eval(slot_pair)) for slot_pair in json.load(open(os.path.join(read_data_dir, f'qadst_slot_pairs.json')))]
                kwargs['slot_pairs'] = slot_pairs
                multi_domain_dials = []
            elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2ANA, DataProcessor.AUG_TYPE_CONCATNANA, DataProcessor.AUG_TYPE_CONCAT2ANA_UNIFORM, \
                                   DataProcessor.AUG_TYPE_10SYN, DataProcessor.AUG_TYPE_15SYN]:
                if args.data_aug == DataProcessor.AUG_TYPE_CONCATNANA:
                    filename = os.path.join(read_data_dir, f'aug{DataProcessor.AUG_TYPE_CONCATNREL}_x100.0/{data_split}_aug_dials_coqr.json')
                else:
                    filename = os.path.join(read_data_dir, f'aug{DataProcessor.AUG_TYPE_CONCAT2REL}_x100.0/{data_split}_aug_dials_coqr.json')
                multi_domain_dials = json.load(open(filename))
                coqr_slot_pairs = {}
                kwargs['coqr_slot_pairs'] = coqr_slot_pairs
            elif args.data_aug in [DataProcessor.AUG_TYPE_10TRUE_AND_SYN, DataProcessor.AUG_TYPE_15TRUE_AND_SYN, 
                                   DataProcessor.AUG_TYPE_10TRUE, DataProcessor.AUG_TYPE_15TRUE, DataProcessor.AUG_TYPE_TRUE_AND_SYN]:
                assert args.dataset_name == 'sgd'
                multi_domain_dials = json.load(open(os.path.join(src_data_dir, f'{data_split}_multi_domain.json')))
                # use concat2rel+anaphora data
                filename = os.path.join(read_data_dir, f'aug{DataProcessor.AUG_TYPE_CONCAT2REL}_x100.0/{data_split}_aug_dials_coqr.json')
                syn_multi_domain_dials = json.load(open(filename))
                kwargs['syn_multi_domain_dials'] = syn_multi_domain_dials
                coqr_slot_pairs = {}
                kwargs['coqr_slot_pairs'] = coqr_slot_pairs
            else:
                assert IndexError("data_aug not in range")
                
            aug_dials = data_processor.data_augmentation(single_domain_dials, multi_domain_dials, args.data_aug_times, **kwargs)

            if data_split=='train' and args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2ANA, DataProcessor.AUG_TYPE_CONCATNANA, 
                                                         DataProcessor.AUG_TYPE_CONCAT2ANA_UNIFORM,
                                                         DataProcessor.AUG_TYPE_10TRUE_AND_SYN, DataProcessor.AUG_TYPE_15TRUE_AND_SYN, 
                                                         DataProcessor.AUG_TYPE_TRUE_AND_SYN]:
                output_filename = os.path.join(write_data_dir, f'coqr_slot_pairs.json')
                json.dump(kwargs['coqr_slot_pairs'], open(output_filename, "w", encoding='utf-8'), indent=2)

            output_filename = os.path.join(write_data_dir, f'{data_split}_aug_dials.json')
            json.dump(aug_dials, open(output_filename, "w", encoding='utf-8'), indent=2)

        table.append({'data_split': f'{data_split} single', 'dialogs': len(single_domain_dials)})
        table.append({'data_split': f'{data_split} augmentation', 'dialogs': len(aug_dials)})

        output_filename = os.path.join(write_data_dir, f'{data_split}_dials.json')
        json.dump(single_domain_dials+aug_dials, open(output_filename, "w", encoding='utf-8'), indent=2)
        
    res = tabulate(table, headers='keys', tablefmt='github')
    with open(f'{write_data_dir}/data_stat.md', 'w', encoding='utf-8') as f:
        print(res, file=f)


def create_coqr_data(args):
    random.seed(42)
    dataset_name = args.dataset_name

    def serialize_input_output(history, question, rewrite, task_type):
        """prepare input and output from original question rewrite dataset"""
        if task_type == 'origin':
            prompt = 'Rewrite the question to remove anaphora and make it self-contained according to the given context.'
            input_str = '{}\n\ncontext: {}\n\nquestion: {}'.format(prompt, '\n'.join(history), question)
            output_str = rewrite

        else:
            prompt = 'Rewrite the question to make the dialog fluent using anaphora according to the given context.'
            if task_type == 'reverse_SDI':
                label_rewrite = label_str(rewrite, question)
            elif task_type == 'reverse_S':
                label_rewrite = label_str(rewrite, question, deletion="sub", insertion="sub")
            elif task_type == 'reverse':
                label_rewrite = rewrite
            elif task_type == 'reverse_infer':
                label_rewrite = rewrite

            input_str = '{}\n\ncontext: {}\n\nquestion: {}'.format(prompt, '\n'.join(history), label_rewrite)
            output_str = question
        return {'input': input_str, 'output': output_str}
    
    if dataset_name=='canard':
        for data_split in ['train', 'dev', 'test']:
            data = json.load(open(os.path.join(args.src_data_dir, f'{data_split}.json')))
            if data_split == 'dev':
                data_split = 'validation'
            data = [(item['History'], item['Question'], item['Rewrite']) for item in data if item['Question']!=item['Rewrite']]

            for task_type in ['origin', 'reverse_SDI']: #'reverse', 'reverse_S']:
                data_dir = os.path.join(args.write_data_dir, task_type)
                os.makedirs(data_dir, exist_ok=True)
                output_filename = os.path.join(data_dir, f'{data_split}.json')
                with open(output_filename, "w", encoding='utf-8') as f:
                    for item in data:
                        sample = serialize_input_output(*item, task_type)
                        if sample is not None:
                            f.write(json.dumps(sample)+'\n')
    
    elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2REL, DataProcessor.AUG_TYPE_CONCATNREL]:
        # create CONCAT2REL data for CoQR: 1) anaphora 2) ellipsis 3) chatgpt for both.
        ontology = load_ontology(dataset_name)
        data_dir = args.write_data_dir
        for data_split in ['train', 'validation']:
            dials = json.load(open(os.path.join(data_dir, f'{data_split}_aug_dials.json')))
            f_anaphora = open(os.path.join(data_dir, f'{data_split}_aug_dials4coqr.json'), 'w', encoding='utf-8')
            # f_ellipsis = open(os.path.join(data_dir, f'{data_split}_aug_dials4elli.json'), 'w', encoding='utf-8')
            # f_chatgpt = open(os.path.join(data_dir, f'{data_split}_aug_dials4gpt.json'), 'w', encoding='utf-8')
            for dial_idx, dial in enumerate(dials):
                if 'target_turn_idx' not in dial:
                    # ignore dials that tgt value is not replaced by src slot value
                    continue

                # 1) anaphora model input
                anaphora_history = []
                target_turn_idxes = dial['target_turn_idx']
                if isinstance(target_turn_idxes, int):
                    target_turn_idxes = [target_turn_idxes]
                for target_turn_idx in target_turn_idxes:
                    replace_state = dial['turns'][target_turn_idx]['replace_state']
                    src_values = []
                    src_slots = []
                    min_source_turn_idx = target_turn_idx
                    for item in replace_state:
                        src_domain, src_slot, src_value = item['source']
                        src_values.append(src_value)
                        src_slots.append(src_slot)
                        src_slot_desc = ontology['domains'][src_domain]['slots'][src_slot]['description']
                        for turn_idx, turn in enumerate(dial['turns']):
                            if 'state' in turn and src_domain in turn['state'] and src_slot in turn['state'][src_domain] and turn['state'][src_domain][src_slot] == src_value:
                                source_turn_idx = turn_idx
                                if source_turn_idx < min_source_turn_idx:
                                    min_source_turn_idx = source_turn_idx
                                break
                    
                        # source domain 3 turns up to the mention of src_value U_{s-2:s}
                        anaphora_history += [dial['turns'][idx]['utterance'] for idx in range(max(0, source_turn_idx-2), source_turn_idx+1)]
                        anaphora_history += [f"The {src_slot_desc} is {src_value}."]
                    
                    # target domain 1 turn before state update U_{t-1:t-1}
                    anaphora_history += [dial['turns'][idx]['utterance'] for idx in range(max(0, target_turn_idx-1), target_turn_idx)]
                    # anaphora_history += [dial['turns'][idx]['utterance'] for idx in range(max(0, min_source_turn_idx-2), target_turn_idx)]
                    utterance = dial['turns'][target_turn_idx]['utterance']
                    rewrite = dial['turns'][target_turn_idx]['utterance4rewrite']
                    ret = serialize_input_output(anaphora_history, utterance, rewrite, 'reverse_infer')
                    f_anaphora.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': target_turn_idx, **ret, 'src_values': src_values, 'src_slots': src_slots})+'\n')

                    # # 2) ellipsis.
                    # f_ellipsis.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': target_turn_idx, 'utterance': utterance, 'src_values': src_values})+'\n')

                    # # 3) chatgpt.
                    # chatgpt_history = [dial['turns'][idx]['utterance'] for idx in range(0, target_turn_idx)]
                    # for item in replace_state:
                    #     for k in ['source', 'target']:
                    #         domain, slot, value = item[k]
                    #         slot_desc = ontology['domains'][domain]['slots'][slot]['description']
                    #         item[k].append(slot_desc)
                    # f_chatgpt.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': target_turn_idx, 'history': chatgpt_history, 'utterance': utterance, 'replace_state': replace_state})+'\n')

            f_anaphora.close()
            # f_ellipsis.close()
            # f_chatgpt.close()
    
    elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2ANA, DataProcessor.AUG_TYPE_CONCATNANA]:
        # create anaphora/ellipsis data for CoQR model training.
        source_data_dir = args.src_data_dir

        data_dir = os.path.join(args.write_data_dir, f'coqr')
        os.makedirs(data_dir, exist_ok=True)
        
        for data_split in ['train', 'validation']:
            syn_dials = json.load(open(os.path.join(args.write_data_dir, f'{data_split}_aug_dials.json')))
            f_coqr = open(os.path.join(data_dir, f'{data_split}.json'), 'w', encoding='utf-8')

            for dial_idx, dial in enumerate(syn_dials):
                remain_turn_idxes = []
                for turn_idx in range(0, len(dial['turns']), 2):
                    turn = dial['turns'][turn_idx]
                    if 'coqr_utterance' in turn:
                        assert turn['coqr_utterance'] == turn['utterance']
                        origin_utt = turn['utterance4rewrite'].replace('<sub> ','').replace(' </sub>', '')
                        history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
                        ret = serialize_input_output(history, turn['utterance'], origin_utt, 'origin')
                        f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
                    else:
                        remain_turn_idxes.append(turn_idx)

                turn_idx = random.choice(remain_turn_idxes)
                history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
                ret = serialize_input_output(history, dial['turns'][turn_idx]['utterance'], dial['turns'][turn_idx]['utterance'], 'origin')
                f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
                
            f_coqr.close()

            # single_domain_dials = json.load(open(os.path.join(source_data_dir, f'{data_split}_single_domain.json')))
            # num_aug_dials = round(len(single_domain_dials) * args.data_aug_times)
            # multi_domain_dials = json.load(open(os.path.join(parent_dir, f'{data_split}_aug_dials.json')))
            # coqr_samples = [json.loads(line) for line in open(os.path.join(parent_dir, f'{data_split}_aug_dials4coqr_filtered_predictions.json'))]

            # selected_dials_idx = set()
            
            # for sample in coqr_samples:
            #     dial_idx, turn_idx = sample['dial_idx'], sample['turn_idx']
            #     selected_dials_idx.add(dial_idx)
            #     dial = multi_domain_dials[dial_idx]
            #     history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
            #     ret = serialize_input_output(history, sample['predictions'], dial['turns'][turn_idx]['utterance'], 'origin')
            #     f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
            #     if len(selected_dials_idx) > (num_aug_dials//6):
            #         break

            # for sample in elli_samples:
            #     dial_idx, turn_idx = sample['dial_idx'], sample['turn_idx']
            #     if dial_idx in selected_dials_idx:
            #         continue
            #     selected_dials_idx.add(dial_idx)
            #     dial = multi_domain_dials[dial_idx]
            #     history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
            #     ret = serialize_input_output(history, sample['new_utterance'], dial['turns'][turn_idx]['utterance'], 'origin')
            #     f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
            #     if len(selected_dials_idx) > (num_aug_dials//3):
            #         break
            
            # for dial_idx, dial in enumerate(multi_domain_dials):
            #     if dial_idx in selected_dials_idx:
            #         continue
            #     selected_dials_idx.add(dial_idx)
            #     turn_idx = random.choice(range(0, len(dial), 2))
            #     history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
            #     ret = serialize_input_output(history, dial['turns'][turn_idx]['utterance'], dial['turns'][turn_idx]['utterance'], 'origin')
            #     f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
            #     if len(selected_dials_idx) > (num_aug_dials//2):
            #         break
            # f_coqr.close()

        data_split = 'test'
        for data_type in ['single', 'multi']:
            infer_dials = json.load(open(os.path.join(source_data_dir, f'test_{data_type}_domain.json')))
            f_coqr = open(os.path.join(data_dir, f'test_{data_type}_domain.json'), 'w', encoding='utf-8')
            for dial_idx, dial in enumerate(infer_dials):
                for turn_idx in range(0, len(dial['turns']), 2):
                    history = [dial['turns'][idx]['utterance'] for idx in range(0, turn_idx)]
                    ret = serialize_input_output(history, dial['turns'][turn_idx]['utterance'], dial['turns'][turn_idx]['utterance'], 'origin')
                    f_coqr.write(json.dumps({'dial_idx': dial_idx, 'turn_idx': turn_idx, **ret})+'\n')
            f_coqr.close()


def create_domain_classifier_data(args):
    random.seed(42)
    read_data_dir = args.read_data_dir
    src_data_dir = args.src_data_dir
    write_data_dir = args.write_data_dir
    context_window_size = args.context_window_size

    for data_split in ['train', 'validation']:
        filename = f'{data_split}_dials.json'
        data = json.load(open(os.path.join(read_data_dir, filename)))
        
        output_filename = os.path.join(write_data_dir, f'{data_split}.json')
        with open(output_filename, "w", encoding='utf-8') as f:
            samples = dials2domain_cls_samples(data, context_window_size)
            f.writelines(samples)
    
    for data_type in ['single', 'multi']:
        filename = f'test_{data_type}_domain.json'
        data = json.load(open(os.path.join(src_data_dir, filename)))

        output_filename = os.path.join(write_data_dir, filename)
        with open(output_filename, "w", encoding='utf-8') as f:
            samples = dials2domain_cls_samples(data, context_window_size)
            f.writelines(samples)


def create_dst_data(args):
    assert args.model_type is not None and args.context_window_size is not None
    read_data_dir = args.read_data_dir
    write_data_dir = args.write_data_dir
    ontology = load_ontology(args.dataset_name)
    full_state = json.load(open(os.path.join(args.src_data_dir, 'full_state.json')))
    data_processor = DataProcessor(args.model_type, args.data_aug, args.context_window_size, ontology, full_state)
    for data_split in ['train', 'validation']:
        filename = os.path.join(read_data_dir, f'{data_split}_dials.json')
        dials = json.load(open(filename))
        output_filename = os.path.join(write_data_dir, f'{data_split}.json')
        with open(output_filename, "w", encoding='utf-8') as f:
            samples = data_processor.dials2samples(dials)
            f.writelines(samples)


# def create_dst_data(dataset_name, data_dir, args):
#     random.seed(42)
#     os.makedirs(data_dir, exist_ok=True)
#     source_data_dir = os.path.dirname(data_dir)
#     table = []
#     ontology = load_ontology(dataset_name)
#     full_state = json.load(open(os.path.join(source_data_dir, 'full_state.json')))
#     data_processor = DataProcessor(args.model_type, args.data_aug, args.context_window_size, ontology, full_state)
#     for data_split in ['train', 'validation']:
#         single_domain_dials = json.load(open(os.path.join(source_data_dir, f'{data_split}_single_domain.json')))

#         # augment multi-domain dialogs from single domain dialogs
#         if args.data_aug == DataProcessor.AUG_TYPE_NONE:
#             # no data augmentation, only original single domain dialogs
#             aug_dials = []
#         else:
#             kwargs = {}
#             if args.data_aug == DataProcessor.AUG_TYPE_REPLACE_TRUE:
#                 multi_domain_dials = json.load(open(os.path.join(source_data_dir, f'{data_split}_multi_domain.json')))
#             elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2, DataProcessor.AUG_TYPE_CONCATN]:
#                 multi_domain_dials = []
#             elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2REL, DataProcessor.AUG_TYPE_CONCATNREL]:
#                 single_domain_dials = json.load(open(os.path.join(source_data_dir, f'{data_split}_single_domain_qa.json')))
#                 slot_pairs = pd.read_csv(os.path.join(source_data_dir, 'qa_slot_pairs.csv'))['slot pair'].to_list()
#                 kwargs['slot_pairs'] = slot_pairs
#                 multi_domain_dials = []
#             elif args.data_aug == DataProcessor.AUG_TYPE_CONCAT2ANA:
#                 filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_coqr_x80.0.json')
#                 if not os.path.exists(filename):
#                     filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_coqr_x{args.data_aug_times*10}.json')
#                 multi_domain_dials = json.load(open(filename))
#             elif args.data_aug == DataProcessor.AUG_TYPE_CONCAT2ELL:
#                 filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_elli_x80.0.json')
#                 if not os.path.exists(filename):
#                     filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_elli_x{args.data_aug_times*10}.json')
#                 multi_domain_dials = json.load(open(filename))
#             elif args.data_aug in [DataProcessor.AUG_TYPE_CONCAT2MIX, DataProcessor.AUG_TYPE_CONCATNMIX]:
#                 if args.data_aug == DataProcessor.AUG_TYPE_CONCATNMIX:
#                     parent_dir = data_dir.replace('model1_context4', 'model0_context100').replace(f'aug{DataProcessor.AUG_TYPE_CONCATNMIX}', f'aug{DataProcessor.AUG_TYPE_CONCATNREL}').replace(f'x{args.data_aug_times}', f'x80.0')
#                 else:
#                     parent_dir = data_dir.replace('model1_context4', 'model0_context100').replace(f'aug{DataProcessor.AUG_TYPE_CONCAT2MIX}', f'aug{DataProcessor.AUG_TYPE_CONCAT2REL}').replace(f'x{args.data_aug_times}', f'x80.0')
#                 if not os.path.exists(parent_dir):
#                     parent_dir = data_dir.replace(f'aug{DataProcessor.AUG_TYPE_CONCAT2MIX}', f'aug{DataProcessor.AUG_TYPE_CONCAT2REL}').replace(f'x{args.data_aug_times}', f'x{args.data_aug_times*10}')
#                 multi_domain_dials = json.load(open(os.path.join(parent_dir, f'{data_split}_aug_dials.json')))
#                 kwargs['coqr_samples'] = [json.loads(line) for line in open(os.path.join(parent_dir, f'{data_split}_aug_dials4coqr_filtered_predictions.json'))]
#                 kwargs['elli_samples'] = [json.loads(line) for line in open(os.path.join(parent_dir, f'{data_split}_aug_dials4elli_filtered_predictions.json'))]
#             elif args.data_aug == DataProcessor.AUG_TYPE_CONCAT2GPT:
#                 filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_chatgpt_x80.0.json')
#                 if not os.path.exists(filename):
#                     filename = os.path.join(source_data_dir, f'{data_split}_aug_dials_chatgpt_x{args.data_aug_times*10}.json')
#                 if not os.path.exists(filename):
#                     continue
#                 multi_domain_dials = json.load(open(filename))
#             else:
#                 assert IndexError("data_aug not in range")
                
#             aug_dials = data_processor.data_augmentation(single_domain_dials, multi_domain_dials, args.data_aug_times, **kwargs)

#             output_filename = os.path.join(data_dir, f'{data_split}_aug_dials.json')
#             json.dump(aug_dials, open(output_filename, "w", encoding='utf-8'), indent=2)

#         output_filename = os.path.join(data_dir, f'{data_split}.json')

#         table.append({'data_split': f'{data_split} single', 'dialogs': len(single_domain_dials)})
#         table.append({'data_split': f'{data_split} augmentation', 'dialogs': len(aug_dials)})
#         with open(output_filename, "w", encoding='utf-8') as f:
#             samples = data_processor.dials2samples(single_domain_dials)
#             f.writelines(samples)
#             table[-2]['samples'] = len(samples)
#             samples = data_processor.dials2samples(aug_dials)
#             f.writelines(samples)
#             table[-1]['samples'] = len(samples)

#         if 'group1' in data_dir or 'group2' in data_dir:
#             # domain num exp, add other domains' single-domain dialog
#             all_single_domain_dials = json.load(open(os.path.join(source_data_dir.replace('group1', 'group0').replace('group2', 'group0'),
#                                                                   f'{data_split}_single_domain.json')))
#             exist_domains = set()
#             for dial in single_domain_dials:
#                 domain = dial['domains'][0]
#                 if domain not in exist_domains:
#                     exist_domains.add(domain)
#             add_single_output_filename = output_filename.replace(data_split, f'{data_split}_all_single')
#             shutil.copy2(output_filename, add_single_output_filename) # original data

#             table.append({'data_split': f'{data_split} added single', 'dialogs': 0, 'samples': 0})
#             # add other single domain dialog
#             with open(add_single_output_filename, "a", encoding='utf-8') as f:
#                 for dial in all_single_domain_dials:
#                     domain = dial['domains'][0]
#                     if domain not in exist_domains:
#                         table[-1]['dialogs'] += 1
#                         samples = data_processor.dials2samples([dial])
#                         f.writelines(samples)
#                         table[-1]['samples'] += len(samples)

#             # add synthesized dialogs that contain other single domains (i.e. one of the domain is not in exist_domains)
#             removed_single_domain_dial_idx = [idx for dial in aug_dials for idx in dial['dials_idx']]
#             assert len(removed_single_domain_dial_idx) == len(set(removed_single_domain_dial_idx))
#             removed_single_domain_dial_idx = set(removed_single_domain_dial_idx)
#             num_aug_dials = round(len(all_single_domain_dials) * 8.0)

#             parent_dir = os.path.join(os.path.dirname(source_data_dir), f'group0/model0_context100_aug{DataProcessor.AUG_TYPE_CONCAT2REL}_x80.0')
#             multi_domain_dials = json.load(open(os.path.join(parent_dir, f'{data_split}_aug_dials.json')))
#             coqr_samples = [json.loads(line) for line in open(os.path.join(parent_dir, f'{data_split}_aug_dials4coqr_filtered_predictions.json'))]
#             elli_samples = [json.loads(line) for line in open(os.path.join(parent_dir, f'{data_split}_aug_dials4elli_filtered_predictions.json'))]

#             selected_dials_idx = set()
#             aug_dials = []
#             for sample in coqr_samples:
#                 dial_idx = sample['dial_idx']
#                 dial = multi_domain_dials[dial_idx]
#                 if all([domain in exist_domains for domain in dial['domains']]):
#                     # skip domain combinations that have true multi-domain data
#                     continue
#                 if any([idx in removed_single_domain_dial_idx for idx in dial['dials_idx']]):
#                     # skip removed single domain dials
#                     continue
#                 selected_dials_idx.add(dial_idx)
#                 dial['turns'][sample['turn_idx']]['utterance'] = sample['predictions']
#                 dial['turns'][sample['turn_idx']]['coqr_utterance'] = sample['predictions']
#                 aug_dials.append(dial)
#                 if len(aug_dials) > (num_aug_dials//6):
#                     break

#             for sample in elli_samples:
#                 dial_idx = sample['dial_idx']
#                 if dial_idx in selected_dials_idx:
#                     continue
#                 dial = multi_domain_dials[dial_idx]
#                 if all([domain in exist_domains for domain in dial['domains']]):
#                     # skip domain combinations that have true multi-domain data
#                     continue
#                 if any([idx in removed_single_domain_dial_idx for idx in dial['dials_idx']]):
#                     # skip removed single domain dials
#                     continue
#                 selected_dials_idx.add(dial_idx)
#                 dial['turns'][sample['turn_idx']]['utterance'] = sample['new_utterance']
#                 dial['turns'][sample['turn_idx']]['elli_utterance'] = sample['new_utterance']
#                 aug_dials.append(dial)
#                 if len(aug_dials) > (num_aug_dials//3):
#                     break
            
#             for dial_idx, dial in enumerate(multi_domain_dials):
#                 if dial_idx in selected_dials_idx:
#                     continue
#                 if all([domain in exist_domains for domain in dial['domains']]):
#                     # skip domain combinations that have true multi-domain data
#                     continue
#                 if any([idx in removed_single_domain_dial_idx for idx in dial['dials_idx']]):
#                     # skip removed single domain dials
#                     continue
#                 aug_dials.append(dial)
#                 if len(aug_dials) > (num_aug_dials//2):
#                     break

#             add_single_mix8x_output_filename = output_filename.replace(data_split, f'{data_split}_all_single_mix8x')
#             shutil.copy2(add_single_output_filename, add_single_mix8x_output_filename)
            
#             table.append({'data_split': f'{data_split} added syn multi', 'dialogs': 0, 'samples': 0})
#             # add other single domain dialog
#             with open(add_single_mix8x_output_filename, "a", encoding='utf-8') as f:
#                 for dial in aug_dials:
#                     table[-1]['dialogs'] += 1
#                     samples = data_processor.dials2samples([dial])
#                     f.writelines(samples)
#                     table[-1]['samples'] += len(samples)

#             output_filename = os.path.join(data_dir, f'{data_split}_aug_dials_syn.json')
#             json.dump(aug_dials, open(output_filename, "w", encoding='utf-8'), indent=2)


#     res = tabulate(table, headers='keys', tablefmt='github')
#     with open(f'{data_dir}/data_stat.md', 'w', encoding='utf-8') as f:
#         print(res, file=f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--task', '-t', help='name of function to call')
    parser.add_argument('--dataset_name', '-d', help='name of the unified dataset')
    parser.add_argument('--group_idx', '-g', type=int, default=None, help='group index for sgd dataset')
    parser.add_argument('--context_window_size', '-c', type=int, default=None, help='how many contextual utterances are considered')
    parser.add_argument('--model_type', '-m', type=int, default=None, help='model type')
    parser.add_argument('--data_aug', '-a', type=int, default=None, help='data augmentation type')
    parser.add_argument('--data_aug_times', '-x', type=float, default=None, help='data augmentation as times of single domain dials')
    parser.add_argument('--read_data_dir', '-r', type=str, default=None, help='read data dir, in addition to src data dir')
    parser.add_argument('--write_data_dir', '-w', type=str, required=True, help='write data dir')
    args = parser.parse_args()
    if args.dataset_name == 'sgd':
        assert args.group_idx is not None
        src_data_dir = os.path.join('data', f'{args.dataset_name}/group{args.group_idx}')
    elif args.dataset_name == 'multiwoz21':
        src_data_dir = os.path.join('data', f'{args.dataset_name}')
    else:
        assert args.dataset_name == 'canard'
        src_data_dir = 'CANARD_Release'

    args.src_data_dir = src_data_dir
    print(args)
    os.makedirs(args.write_data_dir, exist_ok=True)
    if args.task == 'single_domain_qadst':
        create_single_domain_qadst_data(args)
    elif args.task == 'aug_data':
        create_aug_data(args)
    elif args.task == 'coqr':
        create_coqr_data(args)
    elif args.task == 'dst':
        create_dst_data(args)
    elif args.task == 'dom_cls':
        create_domain_classifier_data(args)
    # else:
    #     create_dst_data(args.dataset_name, src_data_dir, args)
