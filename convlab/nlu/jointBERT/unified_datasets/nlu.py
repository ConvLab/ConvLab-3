import logging
import os
import json
import torch
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
import transformers
from convlab.nlu.nlu import NLU
from convlab.nlu.jointBERT.dataloader import Dataloader
from convlab.nlu.jointBERT.jointBERT import JointBERT
from convlab.nlu.jointBERT.unified_datasets.postprocess import recover_intent
from convlab.util.custom_util import model_downloader


class BERTNLU(NLU):
    def __init__(self, mode, config_file, model_file=None):
        assert mode == 'user' or mode == 'sys' or mode == 'all'
        self.mode = mode
        config_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        # print(config['DEVICE'])
        # DEVICE = config['DEVICE']
        DEVICE = 'cpu' if not torch.cuda.is_available() else config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        assert os.path.exists(os.path.join(data_dir, 'intent_vocab.json')), print('Please run preprocess first')

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        logging.info('intent num:' +  str(len(intent_vocab)))
        logging.info('tag num:' + str(len(tag_vocab)))

        if not os.path.exists(output_dir):
            model_downloader(root_dir, model_file)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)

        state_dict = torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE)
        if int(transformers.__version__.split('.')[0]) >= 3 and 'bert.embeddings.position_ids' not in state_dict:
            state_dict['bert.embeddings.position_ids'] = torch.tensor(range(512)).reshape(1, -1).to(DEVICE)

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.use_context = config['model']['context']
        self.context_window_size = config['context_window_size']
        self.dataloader = dataloader
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TreebankWordTokenizer()
        logging.info("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        sentences = self.sent_tokenizer.tokenize(utterance)
        ori_word_seq = [token for sent in sentences for token in self.word_tokenizer.tokenize(sent)]
        ori_tag_seq = [str(('O',))] * len(ori_word_seq)
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context_seq = self.dataloader.tokenizer.encode(' [SEP] '.join(context[-self.context_window_size:]))
            context_seq = context_seq[:510]
        else:
            context_seq = self.dataloader.tokenizer.encode('')
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = self.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        word_seq = word_seq[:510]
        tag_seq = tag_seq[:510]
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        das = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                             batch_data[0][0], batch_data[0][-4])
        dialog_act = []
        for da_type in das:
            for da in das[da_type]:
                dialog_act.append([da['intent'], da['domain'], da['slot'], da.get('value','')])
        return dialog_act


if __name__ == '__main__':
    texts = [
        "I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "I want to leave after 17:15.",
        "Thank you for all the help! I appreciate it.",
        "Please find a restaurant called Nusha.",
        "What is the train id, please? ",
        "I don't care about the price and it doesn't need to have free parking."
    ]
    nlu = BERTNLU(mode='user', config_file='multiwoz21_user.json')
    for text in texts:
        print(text)
        print(nlu.predict(text))
        print()
