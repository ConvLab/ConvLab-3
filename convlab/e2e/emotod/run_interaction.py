import argparse

from convlab.util.analysis_tool.analyzer import Analyzer
from convlab.util.custom_util import evaluate, create_goals, set_seed
from convlab.dialog_agent import PipelineAgent, BiSession
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.e2e.emotod.e2ewrapper import E2EAgentWrapper

from convlab.e2e.emotod.emollama import EMOLLAMAAgent
from convlab.policy.emoUS_v2.langEmoUS import UserPolicy
# from convlab.nlu.jointBERT.multiwoz.nlu import BERTNLU
from convlab.base_models.t5.nlu.nlu import T5NLU
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator

from utils import seed_all

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to llama e2e model')
parser.add_argument('--output_path', type=str, help='path to save dir')
parser.add_argument('--num_dialogues', type=int, default=1, help='number of dialogues to simulate')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--simple', type=bool, help='whether to use simple model')

args = parser.parse_args()

seed = args.seed
seed_all(seed)

sys_nlu = None
sys_dst = None
print('Initialising Emo-LLAMA')
sys_policy = EMOLLAMAAgent(model_file=args.model_path, simple=args.simple)
sys_nlg = None

# sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
sys_agent = E2EAgentWrapper(sys_policy, 'emollama')

# user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', model_file='/home/fengs/projects/pretrained_models/bert_multiwoz_sys_context.zip')
print('Initialising T5NLU')
user_nlu = T5NLU(speaker='system', context_window_size=3, model_name_or_path='/home/shutong/pretrained_models/t5-small-nlu-all-multiwoz21-context3')
user_dst = None
print('Initialising EmoUS_Lang')
emous_configs = {
    'Neutral': 0.95,
    'Satisfied': 0.95,
    'sub_goal_succ': True
}
user_policy = UserPolicy(
    model_checkpoint='/home/shutong/pretrained_models/emous_lang', kwargs=emous_configs)
user_nlg = None

user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

print('Initialising analyzer')
analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

print('Start to analyze')
analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=args.output_path, total_dialog=args.num_dialogues, s=args.seed)

# session = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=MultiWozEvaluator())

# goal_generator = GoalGenerator()
# num_eval_dialogues = args.num_dialogues
# goals = []
# for seed in range(1000, 1000 + num_eval_dialogues):
#     set_seed(seed)
#     goal = create_goals(goal_generator, 1)
#     goals.append(goal[0])

# evaluate(session, num_dialogues=num_eval_dialogues, save_path=args.output_path, goals=goals)