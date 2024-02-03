from convlab.util.analysis_tool.analyzer import Analyzer
from convlab.dialog_agent import PipelineAgent
from convlab.e2e.emotod.e2ewrapper import E2EAgentWrapper

from convlab.e2e.emotod.emotod import EMOTODAgent
from convlab.policy.emoUS.emoUS import UserPolicy
from convlab.nlu.jointBERT.multiwoz.nlu import BERTNLU

sys_nlu = None
sys_dst = None
sys_policy = EMOTODAgent(model_file='/home/fengs/projects/pretrained_models/gpt2_prev_emo')
sys_nlg = None

# sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
sys_agent = E2EAgentWrapper(sys_policy, 'emotod')

user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', model_file='/home/fengs/projects/pretrained_models/bert_multiwoz_sys_context.zip')
user_dst = None
user_policy = UserPolicy(model_checkpoint='/home/fengs/projects/pretrained_models/emous_lang')
user_nlg = None

user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='emotod_gpt2-emous_lang', total_dialog=250)
