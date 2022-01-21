# MultiWOZ Data Preprocessing
Introduction of the data processing procedure in AAAI 2020 paper "Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context". 


## Data Preprocessing

### Delexicalization
Delexicalization is the procedure of replacing specific slot values in dialog utterances by placeholders. The goal is to reduce the surface form language variability and improve model's generalization ability. 

Previous delexicalization methods for MultiWOZ ([Budzianowski et al. 2018](https://arxiv.org/pdf/1810.00278.pdf); [Chen et al. 2019](https://arxiv.org/pdf/1905.12866.pdf)) have a drawback in handling multi-domain slot values. They delexicalize the same slots in different dialog domains such as `phone`, `address`, `name` etc as different tokens, e.g. `<restaurant.phone>` and `<hotel.phone>`, which prevents knowledge sharing through semantically similar slots in different domains and adds extra burdens to the system. We propose the **Domain-Adaptive Delexicalization** to address the problem, by using an identical token to represent the same slot name such as `phone` in different dialog domains. Therefore the expressions in all relevant domains can be used to learn to generate the delexicalized value token. 

Here is an example of different delexicalization strategies:

|||
|---|---|
| Original Utterance | U: I want to find a `cheap` restaurant located in the `west`. Give me the phone numebr please. <br> S: `Thanh Binh` meets your criteria. The phone number is `01223362456`. What else can I help?  |
| Classical Delexicalization | U: I want to find a `<restaurant.pricerange>` restaurant located in the `<restaurant.area>`. <br> S: `<restaurant.name>` meets your criteria. The phone number is `<restaurant.phone>`. What else can I help?  |
| Our Domain-Adaptive <br> Delexicalization | U: I want to find a `<pricerange>` restaurant located in the `<area>`. <br> S: `<name>` meets your criteria. The phone number is `<phone>`. What else can I help?  |


### Tokenization
We use the tokenization tool in spaCy through `spacy.load('en_core_web_sm')`. 

### Normalization
We normalize all the slot names as listed in `ontoloty.py`. 

Functions for dataset cleaning and normalization are in `clean_dataset.py`.

Note that the slot values in databases are also preprocessed via the same procedure. 
