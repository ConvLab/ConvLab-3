from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import html


def preprocess():
    original_data_dir = 'wizard_of_wikipedia'

    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'wow'
    splits = ['train', 'validation', 'test_seen', 'test_unseen']
    dialogues_by_split = {split:[] for split in splits}

    ontology = {'domains': {},
                'intents': {},
                'state': {},
                'dialogue_acts': {
                    "categorical": [],
                    "non-categorical": [],
                    "binary": []
                }}

    for data_split in splits:
        if data_split == 'train':
            filenames = ['train.json']
        elif data_split == 'validation':
            filenames = ['valid_random_split.json', 'valid_topic_split.json']
        elif data_split == 'test_seen':
            filenames = ['test_random_split.json']
        else:
            filenames = ['test_topic_split.json']
        for filename in filenames:
            with open(f'{original_data_dir}/{filename}') as f:
                data = json.load(f)
            for original_dial in tqdm(data, desc=data_split):
                topic = html.unescape(original_dial['chosen_topic'])

                # new dialog
                dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = {
                    'dataset': dataset,
                    'data_split': data_split,
                    'dialogue_id': dialogue_id,
                    'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                    'topic': topic,
                    'turns': []
                }
                dialogues_by_split[data_split].append(dialogue)

                topic2passage = {topic: original_dial['chosen_topic_passage']}

                for original_turn in original_dial['dialog']:
                    speaker = 'system' if 'Wizard' in original_turn['speaker'] else 'user'

                    dialogue['turns'].append({
                        'speaker': speaker,
                        'utterance': original_turn['text'].strip(),
                        'utt_idx': len(dialogue['turns']),
                    })

                    for topic_passage in original_turn['retrieved_passages']:
                        for topic, passage in topic_passage.items():
                            topic = html.unescape(topic)
                            if topic in topic2passage:
                                # topic that already added, add unseen sentences
                                for sen in passage:
                                    if sen not in topic2passage[topic]:
                                        topic2passage[topic].append(sen)
                            else:
                                topic2passage[topic] = passage

                    if speaker == 'system':
                        if len(original_turn['checked_sentence']) == 0:
                            checked_sentence = None
                        else:
                            checked_sentence = list(original_turn['checked_sentence'].values())[0]
                            checked_sentence = None if checked_sentence == 'no_passages_used' else checked_sentence
                        
                        if len(original_turn['checked_passage']) == 0:
                            checked_passage = None
                        else:
                            checked_passage = html.unescape(list(original_turn['checked_passage'].values())[0])
                            # print(topic2passage.keys())
                            checked_passage = None if checked_passage == 'no_passages_used' else topic2passage[checked_passage]

                        if checked_sentence:
                            if not checked_passage or checked_sentence not in checked_passage:
                                # search over retrieved_passages
                                for topic, passage in topic2passage.items():
                                    if checked_sentence in passage:
                                        checked_passage = passage
                                        break
                            assert checked_sentence in checked_passage, print(checked_sentence, checked_passage)

                        dialogue['turns'][-1]['checked_sentence'] = checked_sentence
                        dialogue['turns'][-1]['checked_passage'] = checked_passage

    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test_seen']+dialogues_by_split['test_unseen']
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
