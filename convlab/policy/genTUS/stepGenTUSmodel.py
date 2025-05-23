
import json

import torch
from torch.nn.functional import softmax, one_hot, cross_entropy

from convlab.policy.genTUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.genTUS.token_map import tokenMap
from convlab.policy.genTUS.utils import append_tokens
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, AutoModelForSeq2SeqLM)


class stepGenTUSmodel(torch.nn.Module):
    def __init__(self, model_checkpoint, train_whole_model=True, model_status="evaluation", device="cuda", model_type="encoder_decoder", **kwargs):
        # config = AutoConfig.from_pretrained(model_checkpoint)
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        peft_model_checkpoint = kwargs.get("peft_model_checkpoint", None)
        if peft_model_checkpoint:
            model_type = "llama"
        self.model_type = model_type
        if model_type == "encoder_decoder":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint)

        else:
            from peft import PeftModel
            self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
            self.model = PeftModel.from_pretrained(
                self.model, peft_model_checkpoint)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(device, "device")
        self.model.to(device)
        self.device = device

        self.vocab = len(self.tokenizer)
        self.kg = KnowledgeGraph(
            self.tokenizer, model_type=self.model_type, dataset=kwargs.get("dataset", "multiwoz21"))
        self.action_kg = KnowledgeGraph(
            self.tokenizer, model_type=self.model_type, dataset=kwargs.get("dataset", "multiwoz21"))
        self.token_map = tokenMap(self.tokenizer, model_type=self.model_type)
        # only_action doesn't matter. it is only used for get_log_prob
        self.token_map.default(only_action=True)

        if model_status == "evaluation":
            print("evaluation mode")
            self.model.eval()
            self.model.share_memory()
        elif model_status == "training":
            print("training mode")
            self.model.train()
        else:
            print("unknown model status")

        # if not train_whole_model:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        #     for param in self.model.decoder.layers[-1].fc1.parameters():
        #         param.requires_grad = True
        #     for param in self.model.decoder.layers[-1].fc2.parameters():
        #         param.requires_grad = True

    def get_trainable_param(self):

        return filter(
            lambda p: p.requires_grad, self.model.parameters())

    def get_next_token_logits(self, model_input, generated_so_far):
        input_ids = model_input["input_ids"].to(self.device)
        attention_mask = model_input["attention_mask"].to(self.device)
        generated_so_far = generated_so_far.to(self.device)
        if self.model_type == "encoder_decoder":
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated_so_far,
                return_dict=True)
        else:
            input_ids = torch.cat(
                [input_ids, generated_so_far], -1).to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                return_dict=True)
        return outputs.logits[:, -1, :]

    def get_log_prob(self, s, a, action_mask, prob_mask):
        output = self.model.forward(input_ids=s,
                                    attention_mask=action_mask,
                                    decoder_input_ids=a)
        prob = self._norm_prob(a[:, 1:].long(),
                               output.logits[:, :-1, :],
                               prob_mask[:, 1:, :].long())
        return prob

    def _norm_prob(self, a, prob, mask):
        prob = softmax(prob, -1)
        base = self._base(prob, mask).to(self.device)  # [b, seq_len]
        prob = (prob*one_hot(a, num_classes=self.vocab)).sum(-1)
        prob = torch.log(prob / base)
        pad_mask = a != 1
        prob = prob*pad_mask.float()
        return prob.sum(-1)

    @staticmethod
    def _base(prob, mask):
        batch_size, seq_len, dim = prob.shape
        base = torch.zeros(batch_size, seq_len)
        for b in range(batch_size):
            for s in range(seq_len):
                temp = [prob[b, s, c] for c in mask[b, s, :] if c > 0]
                base[b, s] = torch.sum(torch.tensor(temp))
        return base


if __name__ == "__main__":
    import os
    from convlab.util.custom_util import set_seed
    from convlab.policy.genTUS.stepGenTUS import UserActionPolicy
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_checkpoint = 'convlab/policy/genTUS/experiments/multiwoz21-exp'
    usr = UserActionPolicy(model_checkpoint=model_checkpoint)
    # usr.model.load_state_dict(torch.load(
    #     os.path.join(model_checkpoint, "pytorch_model.bin"), map_location=device))
    # usr.model.eval()

    test_file = "convlab/policy/genTUS/data/goal_status_validation_v1.json"
    data = json.load(open(test_file))
    test_id = 20
    inputs = usr.tokenizer(data["dialog"][test_id]["in"],
                           max_length=400,
                           return_tensors="pt",
                           truncation=True)

    actions = [data["dialog"][test_id]["out"],
               data["dialog"][test_id+100]["out"]]

    for action in actions:
        action = json.loads(action)
        vec = usr.vector.action_vectorize(
            action["action"], s=inputs["input_ids"])

        print({"action": action["action"]})
        print("get_log_prob", usr.model.get_log_prob(
            inputs["input_ids"],
            torch.unsqueeze(vec["vector"], 0),
            inputs["attention_mask"],
            torch.unsqueeze(vec["mask"], 0)))
