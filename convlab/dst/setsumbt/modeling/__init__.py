from transformers import BertConfig, RobertaConfig

from convlab.dst.setsumbt.modeling.setsumbt_nbt import BertSetSUMBT, RobertaSetSUMBT, EnsembleSetSUMBT
from convlab.dst.setsumbt.modeling.ontology_encoder import OntologyEncoder
from convlab.dst.setsumbt.modeling.temperature_scheduler import LinearTemperatureScheduler
from convlab.dst.setsumbt.modeling.trainer import SetSUMBTTrainer
from convlab.dst.setsumbt.modeling.tokenization import SetSUMBTTokenizer

class BertSetSUMBTTokenizer(SetSUMBTTokenizer('bert')): pass
class RobertaSetSUMBTTokenizer(SetSUMBTTokenizer('roberta')): pass

SetSUMBTModels = {
    'bert': (BertSetSUMBT, OntologyEncoder('bert'), BertConfig, BertSetSUMBTTokenizer),
    'roberta': (RobertaSetSUMBT, OntologyEncoder('roberta'), RobertaConfig, RobertaSetSUMBTTokenizer),
    'ensemble': (EnsembleSetSUMBT, None, None, None)
}
