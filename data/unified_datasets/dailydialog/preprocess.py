from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
from collections import Counter
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

topic_map = {
    1: "Ordinary Life", 
    2: "School Life", 
    3: "Culture & Education",
    4: "Attitude & Emotion", 
    5: "Relationship", 
    6: "Tourism", 
    7: "Health", 
    8: "Work", 
    9: "Politics", 
    10: "Finance"
}

act_map = {
    1: "inform", 
    2: "question", 
    3: "directive", 
    4: "commissive"
}

emotion_map = {
    0: "no emotion", 
    1: "anger", 
    2: "disgust", 
    3: "fear", 
    4: "happiness", 
    5: "sadness", 
    6: "surprise"
}

def preprocess():
    original_data_dir = 'ijcnlp_dailydialog'
    new_data_dir = 'data'

    if not os.path.exists(original_data_dir):
        original_data_zip = 'ijcnlp_dailydialog.zip'
        if not os.path.exists(original_data_zip):
            raise FileNotFoundError(f'cannot find original data {original_data_zip} in dailydialog/, should manually download ijcnlp_dailydialog.zip from http://yanran.li/files/ijcnlp_dailydialog.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'dailydialog'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}
    dial2topics = {}
    with open(os.path.join(original_data_dir, 'dialogues_text.txt')) as dialog_file, \
        open(os.path.join(original_data_dir, 'dialogues_topic.txt')) as topic_file:
        for dialog, topic in zip(dialog_file, topic_file):
            topic = int(topic.strip())
            dialog = dialog.replace(' __eou__ ', ' ')
            if dialog in dial2topics:
                dial2topics[dialog].append(topic)
            else:
                dial2topics[dialog] = [topic]

    global topic_map, act_map, emotion_map

    ontology = {'domains': {x:{'description': '', 'slots': {}} for x in topic_map.values()},
                'intents': {x:{'description': ''} for x in act_map.values()},
                'state': {},
                'dialogue_acts': {
                    "categorical": [],
                    "non-categorical": [],
                    "binary": {}
                }}

    detokenizer = TreebankWordDetokenizer()

    for data_split in splits:
        archive = ZipFile(os.path.join(original_data_dir, f'{data_split}.zip'))
        with archive.open(f'{data_split}/dialogues_{data_split}.txt') as dialog_file, \
            archive.open(f'{data_split}/dialogues_act_{data_split}.txt') as act_file, \
            archive.open(f'{data_split}/dialogues_emotion_{data_split}.txt') as emotion_file:
            for dialog_line, act_line, emotion_line in tqdm(zip(dialog_file, act_file, emotion_file)):
                if not dialog_line.strip():
                    break
                utts = dialog_line.decode().split("__eou__")[:-1]
                acts = act_line.decode().split(" ")[:-1]
                emotions = emotion_line.decode().split(" ")[:-1]
                assert (len(utts) == len(acts) == len(emotions)), "Different turns btw dialogue & emotion & action"

                topics = dial2topics[dialog_line.decode().replace(' __eou__ ', ' ')]
                topic = Counter(topics).most_common(1)[0][0]
                domain = topic_map[topic]
                
                dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = {
                    'dataset': dataset,
                    'data_split': data_split,
                    'dialogue_id': dialogue_id,
                    'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                    'domains': [domain],
                    'turns': []
                }

                for utt, act, emotion in zip(utts, acts, emotions):
                    speaker = 'user' if len(dialogue['turns']) % 2 == 0 else 'system'
                    intent = act_map[int(act)]
                    emotion = emotion_map[int(emotion)]
                    # re-tokenize
                    utt = ' '.join([detokenizer.detokenize(word_tokenize(s)) for s in sent_tokenize(utt)])
                    # replace with common apostrophe
                    utt = utt.replace(' ’ ', "'")
                    # add space after full-stop
                    utt = re.sub('\.(?!com)(\w)', lambda x: '. '+x.group(1), utt)

                    dialogue['turns'].append({
                        'speaker': speaker,
                        'utterance': utt.strip(),
                        'utt_idx': len(dialogue['turns']),
                        'dialogue_acts': {
                            'binary': [{
                                'intent': intent, 
                                'domain': '', 
                                'slot': ''
                            }],
                            'categorical': [],
                            'non-categorical': [],
                        },
                        'emotion': emotion,
                    })

                    ontology["dialogue_acts"]['binary'].setdefault((intent, '', ''), {})
                    ontology["dialogue_acts"]['binary'][(intent, '', '')][speaker] = True

                dialogues_by_split[data_split].append(dialogue)

    ontology["dialogue_acts"]['binary'] = sorted([str({'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent':da[0],'domain':da[1], 'slot':da[2]}) for da, speakers in ontology["dialogue_acts"]['binary'].items()])
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(original_data_dir)
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
