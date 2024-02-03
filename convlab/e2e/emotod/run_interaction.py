from convlab.util.analysis_tool.analyzer import Analyzer
from convlab.dialog_agent import PipelineAgent
from convlab.e2e.emotod.e2ewrapper import E2EAgentWrapper

from convlab.e2e.emotod.emotod import EMOTODAgent
from convlab.policy.emoUS_v2.langEmoUS import UserPolicy
# from convlab.nlu.jointBERT.multiwoz.nlu import BERTNLU
from convlab.base_models.t5.nlu.nlu import T5NLU

from utils import seed_all

seed_all(1)

sys_nlu = None
sys_dst = None
sys_policy = EMOTODAgent(model_file='/home/fengs/projects/pretrained_models/gpt2_prev_emo')
sys_nlg = None

# sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
sys_agent = E2EAgentWrapper(sys_policy, 'emotod')

# user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', model_file='/home/fengs/projects/pretrained_models/bert_multiwoz_sys_context.zip')
user_nlu = T5NLU(speaker='system', context_window_size=3, model_name_or_path='/home/fengs/projects/pretrained_models/t5-small-nlu-all-multiwoz21-context3')
user_dst = None
user_policy = UserPolicy(model_checkpoint='/home/fengs/projects/pretrained_models/emous_lang')
user_nlg = None

user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='emotod_gpt2-emous_lang', total_dialog=250)
