import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


def extract_action_distributions_across_seeds(algorithm_dir_path):
    '''
    We extract the information directly from the train_INFO.log file. An evaluation step has either of the two forms:

    Evaluating at start - 2022-11-03-08-53-38------------------------------------------------------------
    Complete: 0.636+-0.02, Success: 0.51+-0.02, Success strict: 0.432+-0.02, Average......
    **OR**
    Evaluating after Dialogues: 1000 - 2022-11-03-09-18-42------------------------------------------------------------
    Complete: 0.786+-0.02, Success: 0.686+-0.02, Success strict: 0.634+-0.02, Average Return: 24.42.......
    '''

    seed_dir_paths = [f.path for f in os.scandir(algorithm_dir_path) if f.is_dir()]
    seed_dir_names = [f.name for f in os.scandir(algorithm_dir_path) if f.is_dir()]

    # dict below will have the form {0: {book: [], inform: [], ..}, 1000: {book: [], inform: [], ..}, ...}
    # where 0 and 1000 are evaluation steps and the list will be as long as the number of seeds used
    distribution_per_step_dict = {}

    for seed_dir_name, seed_dir_path in zip(seed_dir_names, seed_dir_paths):

        with open(os.path.join(seed_dir_path, 'logs', 'train_INFO.log'), "r") as f:
            evaluation_found = False
            num_dialogues = 0
            for line in f:
                line = line.strip()
                if "evaluating at" in line.lower() or "evaluating after" in line.lower():
                    evaluation_found = True
                    if not "at start" in line.lower():
                        num_dialogues = int(line.split(" ")[3])
                    continue
                if evaluation_found:
                    # extracts the strings "book action: 0.3", "inform action: 0.4" ....
                    action_distribution_string = [a for a in line.split(", ")
                                                  if "actions" in a.lower() and "average actions" not in a.lower()]

                    if num_dialogues in distribution_per_step_dict:
                        for action_string in action_distribution_string:
                            action = action_string.lower().split(" ")[0]
                            distribution = float(action_string.lower().split(": ")[-1])
                            if action in distribution_per_step_dict[num_dialogues]:
                                distribution_per_step_dict[num_dialogues][action].append(distribution)
                            else:
                                distribution_per_step_dict[num_dialogues][action] = [distribution]
                    else:
                        distribution_per_step_dict[num_dialogues] = {}
                        for action_string in action_distribution_string:
                            action = action_string.lower().split(" ")[0]
                            distribution = float(action_string.lower().split(": ")[-1])
                            distribution_per_step_dict[num_dialogues][action] = [distribution]

                    evaluation_found = False

    # aggregate across seeds
    for step in distribution_per_step_dict:
        for action in distribution_per_step_dict[step]:
            mean = round(np.mean(distribution_per_step_dict[step][action]), 2)
            std_error = round(np.std(distribution_per_step_dict[step][action]), 2) / np.sqrt(len(seed_dir_paths))
            distribution_per_step_dict[step][action] = {"mean": mean, "error": std_error}

    return distribution_per_step_dict


def plot_distributions(dir_path, alg_maps, output_dir):
    clrs = sns.color_palette("husl", len(alg_maps))

    alg_paths = [os.path.join(dir_path, alg_map['dir']) for alg_map in alg_maps]
    action_distributions = [extract_action_distributions_across_seeds(path) for path in alg_paths]
    possible_actions = action_distributions[0][0].keys()

    create_bar_plots(action_distributions, alg_maps, possible_actions, output_dir)

    for action in possible_actions:
        plt.clf()
        for i, alg_distribution in enumerate(action_distributions):
            steps = alg_distribution.keys()
            try:
                distributions = [alg_distribution[step][action] for step in steps]

                with sns.axes_style("darkgrid"):
                    plt.plot(steps, distributions, c=clrs[i], label=f"{alg_maps[i]['legend']}")
                    plt.yticks(np.arange(0, 1, 0.1))
                    plt.xticks(fontsize=7, rotation=0)
            except:
                # catch if an algorithm does not have a specific action
                pass

        plt.xlabel('Training dialogues')
        plt.ylabel(f"{action} action probability")
        plt.title(f"{action} action probability")
        plt.legend(fancybox=True, shadow=False, ncol=1, loc='upper left')
        plt.savefig(output_dir + f'/{action}_probability.pdf', bbox_inches='tight')


def create_bar_plots(action_distributions, alg_maps, possible_actions, output_dir):

    max_step = max(action_distributions[0].keys())
    final_distributions = [distribution[max_step] for distribution in action_distributions]

    df_list = []
    for action in possible_actions:
        action_list = [action]
        for distribution in final_distributions:
            action_list.append(distribution[action])
        df_list.append(action_list)

    df = pd.DataFrame(df_list, columns=['Probabilities'] + [alg_map["legend"] for alg_map in alg_maps])

    fig = df.plot(x='Probabilities', kind='bar', stacked=False, title='Final Action Distributions',
                  rot=0, grid=True).get_figure()
    plt.yticks(np.arange(0, 1, 0.1))
    fig.savefig(os.path.join(output_dir, "final_action_probabilities.pdf"))
