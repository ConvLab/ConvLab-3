import json
import numpy as np
import itertools
import matplotlib.pyplot as plt

possible_emotions = ['neutral', 'satisfied', 'dissatisfied', 'abusive']


def convolve(time_sequence, window_size=500):
    kernel = np.ones(window_size) / window_size
    average_per_step = np.convolve(time_sequence, kernel, mode='valid')
    return average_per_step


def extract_emotions(path, window_size=1000, save_path=None):
    '''
    Extracts the emotions from the emotion_temperature_logs.json file
    :param path: path to the emotion_temperature_logs.json file
    :param window_size: size of the window to extract the emotions
    Plots the emotion distribution over the window
    '''

    emotion_counter = dict()

    with open(path, "r") as f:
        file = json.load(f)
    file = list(itertools.chain(*file))
    emotions = [x[0] for x in file]
    for emotion in possible_emotions:
        emotion_binaries = [1 if x == emotion else 0 for x in emotions]
        emotion_counter[emotion] = convolve(emotion_binaries, window_size)

    # plot the emotion counter values
    for emotion in emotion_counter:
        plt.plot(emotion_counter[emotion], label=emotion)
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


if __name__ == "__main__":
    path = "emotion_ddpt_exps/baseline/experiment_2023-04-13-14-13-45/save/emotion_temperature_logs.json"
    path_2 = "emotion_ddpt_exps/emotion_reward_explore/experiment_2023-04-13-14-37-25/save/emotion_temperature_logs.json"
    extract_emotions(path, save_path="test.pdf")