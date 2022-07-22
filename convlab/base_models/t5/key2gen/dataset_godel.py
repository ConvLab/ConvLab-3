# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data processing for vanilla generator"""

import json
import datasets
from convlab.base_models.t5.key2gen.features import FEATURES
from copy import deepcopy


class GodelDataset(datasets.GeneratorBasedBuilder):
    """Dataset for vanilla generator (e.g., t5)"""

    VERSION = datasets.Version("1.18.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="nlg", version=VERSION, description="DA grounded generation task"),
        datasets.BuilderConfig(name="kvret", version=VERSION, description="KB grounded generation task"),
        datasets.BuilderConfig(name="opendialkg", version=VERSION, description="KG grounded generation task"),
        datasets.BuilderConfig(name="wow", version=VERSION, description="Passage grounded generation task"),
        datasets.BuilderConfig(name="personachat", version=VERSION, description="Persona grounded generation task"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=f"Vanilla Dataset for {self.config.description}",
            features=datasets.Features(deepcopy(FEATURES[self.config.name]))
        )

    def _split_generators(self, dl_manager):
        generators = []
        if "train" in self.config.data_files:
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": self.config.data_files["train"][0],
                    "split": "train",
                },
            ))
        if "validation" in self.config.data_files:
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": self.config.data_files["validation"][0],
                    "split": "validation",
                },
            ))
        if "test" in self.config.data_files:
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": self.config.data_files["test"][0],
                    "split": "test",
                },
            ))
            
        return generators

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                item = json.loads(row)
                if self.config.name == "nlg":
                    knowledge = item["knowledge"]
                    triples = []
                    for da_type in knowledge:
                        for da in knowledge[da_type]:
                            intent, domain, slot, value = da["intent"], da["domain"], da["slot"], da.get("value", "")
                            if 'start' in da:
                                da.pop('start')
                                da.pop('end')
                            intent_domain = f"{intent}-{domain}"
                            triples.append([intent_domain])
                            if len(slot) > 0:
                                triples[-1].append(slot)
                            if len(value) > 0:
                                triples[-1].append(value)
                    knowledge_seq = "| {} |".format(" | ".join([" : ".join(da_keywords) for da_keywords in triples]))
                    
                elif self.config.name == "kvret":
                    knowledge = {"schedule": [], "weather": [], "navigate": []}
                    triples = []
                    for domain, db_items in item["knowledge"].items():
                        knowledge[domain] = db_items
                        for db_item in db_items:
                            entity = db_item["entity"]
                            for db_key, db_value in db_item.items():
                                if db_key == "entity":
                                    continue
                                triples.append([entity, db_key, db_value])
                    knowledge_seq = "| {} |".format(" | ".join([" : ".join(triple) for triple in triples]))

                elif self.config.name == "opendialkg":
                    knowledge = item["knowledge"]
                    knowledge_seq = "| {} |".format(" | ".join([" : ".join(triple) for triple in item["knowledge"]]))
                
                elif self.config.name in ["wow", "personachat"]:
                    knowledge = item["knowledge"]
                    try:
                        knowledge_seq = "| {} |".format(" | ".join(item["knowledge"]))
                    except:
                        print([knowledge])
                        raise
                
                context = " EOS ".join([turn[1] for turn in item["context"]])
                context_knowledge = context + ' <|Knowledge|> \n\n' + knowledge_seq + ' => '
                
                yield key, {
                    "context+knowledge": context_knowledge,
                    "response": item["response"],
                    "knowledge": knowledge,
                }
