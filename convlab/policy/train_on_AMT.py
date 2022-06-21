import pickle

from convlab.policy.vtrace_rnn_action_embedding import VTRACE_RNN
from convlab.policy.vtrace_rnn_action_embedding.train import evaluate


def avg_token(utterances, min_avg=1.2):
    token = []
    for turn in utterances:
        if turn != '':
            token.append(len(turn.split()))
    avg_len = (sum(token)) / len(token)
    if avg_len < min_avg:
        return False
    return True


def word_token_ratio(utterances, ratio=0.2):
    unique_word = []
    token_num = 0
    for turn in utterances:
        if turn != '':
            for token in turn.split():
                token_num += 1
                if token not in unique_word:
                    unique_word.append(token)
    if len(unique_word) / token_num < ratio:
        return False
    return True


def process_ppo_data(path):
    with open(path, 'rb') as file:
        memory = pickle.load(file)

    feedback = memory.feedback
    states = memory.states
    actions = memory.actions
    rewards = memory.rewards
    probs = memory.action_probs
    sys_output = memory.sys_outputs
    utterances = memory.utterances

    new_actions = []

    for i, action_list in enumerate(actions):
        new_action_list = []
        for j, action in enumerate(action_list):
            new_action_list.append({"action_index": action, "mu": probs[i][j], "mask": [0] * (len(action)+2)})
        new_actions.append(new_action_list)

    return feedback, states, new_actions, rewards, sys_output, utterances


memory_path = "convlab/policy/best_policies/AMT_memory_for_prefilling.pkl"
amt_data_path = "AMT_Experiments_AMT_REAL_VTRACE_RNN_AGENT_2020-09-02-15-10-49_AMT_memory.pkl"
ppo_data_path = "AMT_Experiments_AMT_REAL_PPO_RNN_AGENT_2020-09-02-15-22-25_AMT_memory.pkl"

with open(amt_data_path, 'rb') as file:
    memory = pickle.load(file)

utterances = memory.utterances
feedback = memory.feedback
states = memory.states
actions = memory.actions
rewards = memory.rewards
sys_output = memory.sys_outputs

print("Length of Dataset: ", len(feedback))

ppo_feedback, ppo_states, ppo_actions, ppo_rewards, ppo_outputs, ppo_utterances = process_ppo_data(ppo_data_path)

memory.feedback.extend(ppo_feedback)
memory.states.extend(ppo_states)
memory.actions.extend(ppo_actions)
memory.rewards.extend(ppo_rewards)
memory.sys_outputs.extend(ppo_outputs)
memory.utterances.extend(ppo_utterances)

with open('AMT_memory_merged.pkl', 'wb') as output:
    pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)

delete_indices = []
for i in range(0, len(states)):
    found = False
    if len(states[i]) == 0 or len(actions[i]) == 0 or len(rewards[i]) == 0:
        delete_indices.append(i)
        continue
    if len(utterances[i]) == 0:
        delete_indices.append(i)
        continue
    for ut in utterances[i]:
        if len(ut) == 0:
            delete_indices.append(i)
            found = True
            break
    if found:
        continue
    if len(utterances[i]) < 4:
        delete_indices.append(i)
        continue
    if not word_token_ratio(utterances[i]):
        delete_indices.append(i)
        continue
    if not avg_token(utterances[i]):
        delete_indices.append(i)
        continue

list(set(delete_indices))

for n, i in enumerate(delete_indices):

    if n<100:
        continue

    print(f"Dialog {i}" + "-"*80)
    fb = feedback[i]
    utt = utterances[i]
    sys_out = sys_output[i]

    for u, s in zip(utt, sys_out):
        print("User: ", u)
        print("System", s)
    print("Feedback: ", fb)
    print(n)
    if n == 200:
        break


print("Dialogues before cleaning: ", len(states))
print("Dialogues that will be cleaned: ", len(delete_indices))

states = [states[i] for i in range(len(states)) if i not in delete_indices]
actions = [actions[i] for i in range(len(actions)) if i not in delete_indices]
rewards = [rewards[i] for i in range(len(rewards)) if i not in delete_indices]
feedback = [feedback[i] for i in range(len(feedback)) if i not in delete_indices]

num_good = 0
succesful = 0
for reward, fb in zip(rewards, feedback):
    #print("reward: ", reward)
    #print("feedback: ", fb)
    if -1 not in reward:
        num_good += 1
    if fb:
        succesful += 1

print(f"We have {num_good} good out of {len(rewards)}")
print(f"We have {succesful} successful out of {len(feedback)}")

new_rewards = []
for i, reward in enumerate(rewards):
    new_rewards.append([-1] * len(reward))
    if feedback[i]:
        new_rewards[-1][-1] += 80
    else:
        new_rewards[-1][-1] += -40

rewards = new_rewards
print("Length of Dataset after cleaning: ", len(states))
#print(rewards)


model_path = "convlab/policy/best_policies/RNN_supervised"
policy = VTRACE_RNN(is_train=True, seed=0, shrink=True, noisy=True)
policy.load(model_path)

try:
    policy.prefill_buffer_from_amt(memory_path)
    print("Successfully prefilled buffer with AMT data")
except:
    print("Could not prefill buffer with AMT data")

offset = 0
for i in range(offset):
    policy.update_memory(None, states[i], actions[i], rewards[i])

update_round = 20
eval_freq = 400

for i in range(len(feedback) - offset):

    if i % eval_freq == 0 and i != 0:
        print("Evaluating")
        evaluate(policy_sys=policy)

    policy.update_memory(None, states[offset + i], actions[offset + i], rewards[offset + i])
    if i % update_round == 0:
        print("Updating")
        for k in range(1):
            policy.update()
