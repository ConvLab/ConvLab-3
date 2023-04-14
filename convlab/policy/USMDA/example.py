from datasets import Dataset
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

raw_data = {
    "train": [{"label": 0, "text": "hi how are you"},
              {"label": 1, "text": "i'm fine thank you"}, ],
    "test": [{"label": 0, "text": "hi how are you"},
             {"label": 1, "text": "i'm fine thank you"}, ]}
data = {}
for x in raw_data:
    data[x] = Dataset.from_list(raw_data[x])


def tokenize_function(examples):
    print(examples)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


t = data["train"].map(tokenize_function, batched=True)

print(t)
