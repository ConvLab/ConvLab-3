import pickle, json
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--path", type=str, default="", help="path of model to load")
args = parser.parse_args()

with open(args.path, 'rb') as file:
    memory = pickle.load(file)

feedback = memory.feedback
utterances = memory.utterances
sys_output = memory.sys_outputs
task_ids = memory.task_id

#with open("convlab/dialcrowd_server/task.out", 'r') as f:
#    tasks = json.load(f)

print("TASK IDS", task_ids)

for i in range(40, 80):
    print(f"Dialog {i}" + "-"*80)
    fb = feedback[i]
    utt = utterances[i]
    sys_out = sys_output[i]
    id = task_ids[i]

    for u, s in zip(utt, sys_out):
        print("User: ", u)
        print("System", s)
    print("Feedback: ", fb)
    print("Task ID: ", id)

window = 500
print(f"Number of feedbacks: {len(feedback)}")
ts = pd.Series(feedback)

#moving_average = np.convolve(np.array(feedback), np.ones((window,))/window, mode='valid')
moving_average = ts.rolling(window=500).mean().plot(style='k')
moving_std = ts.rolling(window=500).std().plot(style='b')
print("MOVING AVERAGE:")
print(moving_average)

#plt.plot(np.arange(len(moving_average)), moving_average)
plt.show()
