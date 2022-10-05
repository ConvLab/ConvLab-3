import os, json, logging
import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel
from convlab.policy.vtrace_DPT.transformer_model.noisy_linear import NoisyLinear
from convlab.policy.vtrace_DPT.create_descriptions import create_description_dicts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NodeEmbedderRoberta(nn.Module):
    '''
    Class to build node embeddings
    Nodes have attributes: Description, value and node type that are used for building embedding
    '''

    def __init__(self, projection_dim, freeze_roberta=True, use_pooled=False, max_length=25, roberta_path="",
                 description_dict=None, semantic_descriptions=True, mean=False, dataset_name="multiwoz21"):
        super(NodeEmbedderRoberta, self).__init__()

        self.dataset_name = dataset_name
        self.max_length = max_length
        self.description_size = 768
        self.projection_dim = projection_dim
        self.feature_projection = torch.nn.Linear(2 * self.description_size, projection_dim).to(DEVICE)
        #self.feature_projection = NoisyLinear(2 * self.description_size, projection_dim).to(DEVICE)
        self.value_embedding = torch.nn.Linear(1, self.description_size).to(DEVICE)

        self.semantic_descriptions = semantic_descriptions
        self.init_description_dict()

        self.description2idx = dict((descr, i) for i, descr in enumerate(self.description_dict))
        self.idx2description = dict((i, descr) for descr, i in self.description2idx.items())
        self.use_pooled = use_pooled
        self.mean = mean
        self.embedded_descriptions = None

        if roberta_path:
            embedded_descriptions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      f'embedded_descriptions_{self.dataset_name}.pt')
            if os.path.exists(embedded_descriptions_path):
                self.embedded_descriptions = torch.load(embedded_descriptions_path).to(DEVICE)
            else:
                logging.info(f"Loading Roberta from path {roberta_path}")
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.roberta_model = RobertaModel.from_pretrained(roberta_path).to(DEVICE)

        else:
            embedded_descriptions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      f'embedded_descriptions_base_{self.dataset_name}.pt')
            if os.path.exists(embedded_descriptions_path):
                self.embedded_descriptions = torch.load(embedded_descriptions_path).to(DEVICE)
            else:
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.roberta_model = RobertaModel.from_pretrained("roberta-base").to(DEVICE)

        if self.embedded_descriptions is None:
            if freeze_roberta:
                for param in self.roberta_model.parameters():
                    param.requires_grad = False
            #We embed descriptions beforehand and only make a lookup for better efficiency
            self.form_embedded_descriptions()
            torch.save(self.embedded_descriptions, embedded_descriptions_path)

        logging.info(f"Embedding semantic descriptions: {semantic_descriptions}")
        logging.info(f"Embedded descriptions successfully. Size: {self.embedded_descriptions.size()}")
        logging.info(f"Data set used for descriptions: {dataset_name}")

    def form_embedded_descriptions(self):

        self.embedded_descriptions = self.embed_sentences(
            [self.description_dict[self.idx2description[i]] for i in range(len(self.description_dict))])

    def description_2_idx(self, kg_info):
        embedded_descriptions_idx = torch.Tensor([self.description2idx[node["description"]] for node in kg_info])\
            .long()
        return embedded_descriptions_idx

    def forward(self, description_idx, values):

        #embedded_descriptions = torch.stack(
        #    [self.embedded_descriptions[idx] for idx in description_idx]).to(DEVICE)
        embedded_descriptions = self.embedded_descriptions[description_idx]
        description_value_tensor = torch.cat((self.value_embedding(values),
                                                  embedded_descriptions), dim=-1).to(DEVICE)

        node_embedding = self.feature_projection(description_value_tensor).to(DEVICE)

        return node_embedding

    def embed_sentences(self, sentences):

        tokenized = [self.tokenizer.encode_plus(sen, add_special_tokens=True, max_length=self.max_length,
                                                padding='max_length') for sen in sentences]

        input_ids = torch.Tensor([feat['input_ids'] for feat in tokenized]).long().to(DEVICE)
        attention_mask = torch.Tensor([feat['attention_mask'] for feat in tokenized]).long().to(DEVICE)

        roberta_output = self.roberta_model(input_ids, attention_mask)
        output_states = roberta_output.last_hidden_state
        pooled = roberta_output.pooler_output

        if self.mean:
            length_mask = torch.Tensor([[1 if id_ != 1 else 0 for id_ in ids] for ids in input_ids]).unsqueeze(-1)\
                .to(DEVICE)
            output = (output_states * length_mask).sum(dim=1) / length_mask.sum(dim=1)
        else:
            output = pooled if self.use_pooled else output_states[:, 0, :]

        return output

    def init_description_dict(self):

        create_description_dicts(self.dataset_name)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.semantic_descriptions:
            path = os.path.join(root_dir, f'descriptions/semantic_information_descriptions_{self.dataset_name}.json')
        else:
            path = os.path.join(root_dir, 'information_descriptions.json')
        with open(path, "r") as f:
            self.description_dict = json.load(f)

