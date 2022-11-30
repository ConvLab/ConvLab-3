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

    seed_dir_paths = [f.path for f in os.scandir(
        algorithm_dir_path) if f.is_dir()]
    seed_dir_names = [f.name for f in os.scandir(
        algorithm_dir_path) if f.is_dir()]

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
                            distribution = float(
                                action_string.lower().split(": ")[-1])
                            if action in distribution_per_step_dict[num_dialogues]:
                                distribution_per_step_dict[num_dialogues][action].append(
                                    distribution)
                            else:
                                distribution_per_step_dict[num_dialogues][action] = [
                                    distribution]
                    else:
                        distribution_per_step_dict[num_dialogues] = {}
                        for action_string in action_distribution_string:
                            action = action_string.lower().split(" ")[0]
                            distribution = float(
                                action_string.lower().split(": ")[-1])
                            distribution_per_step_dict[num_dialogues][action] = [
                                distribution]

                    evaluation_found = False

    return distribution_per_step_dict


def plot_distributions(dir_path, alg_maps, output_dir, fill_between=0.3, fontsize=16, font="Times New Roman",
                       figsize=(12, 8), facecolor='#E6E6E6'):
    plt.rcParams["font.family"] = font
    clrs = sns.color_palette("husl", len(alg_maps))

    alg_paths = [os.path.join(dir_path, alg_map['dir'])
                 for alg_map in alg_maps]
    action_distributions = [
        extract_action_distributions_across_seeds(path) for path in alg_paths]
    possible_actions = action_distributions[0][0].keys()

    create_bar_plots(action_distributions, alg_maps,
                     possible_actions, output_dir,
                     fontsize, figsize, facecolor)

    for action in possible_actions:
        plt.clf()
        plt.figure(figsize=figsize)
        plt.gca().patch.set_facecolor(facecolor)
        plt.grid(color='w', linestyle='solid', alpha=0.5)

        largest_max = 0
        smallest_min = 1
        for i, alg_distribution in enumerate(action_distributions):
            steps = alg_distribution.keys()
            try:
                distributions = np.array(
                    [alg_distribution[step][action] for step in steps])
                # length = num_Evaluations * num_seeds
                mean, std_dev = np.mean(distributions, axis=1), np.std(
                    distributions, axis=1)
                seeds_used = distributions.shape[1]
                std_error = std_dev / np.sqrt(seeds_used)

                # with sns.axes_style("darkgrid"):
                plt.plot(steps, mean, c=clrs[i],
                         label=f"{alg_maps[i]['legend']}")
                plt.fill_between(
                    steps, mean - std_error,
                    mean + std_error, alpha=fill_between, facecolor=clrs[i])

                largest_max = mean.max() if mean.max() > largest_max else largest_max
                smallest_min = mean.min() if mean.min() < smallest_min else smallest_min

            except Exception as e:
                # catch if an algorithm does not have a specific action
                print(e)
        print(action)
        if round((largest_max - smallest_min) / 10.0, 2) > 0:
            plt.gca().yaxis.set_major_locator(plt.MultipleLocator(
                round((largest_max - smallest_min) / 10.0, 2)))
        plt.xticks(fontsize=fontsize-4, rotation=0)
        plt.yticks(fontsize=fontsize-4)
        plt.xlabel('Training Dialogues', fontsize=fontsize)
        plt.ylabel(f"{action.title()} Intent Probability", fontsize=fontsize)
        plt.legend(fancybox=True, shadow=False, ncol=1, loc='best')
        plt.savefig(
            output_dir + f'/{action}_probability.pdf', bbox_inches='tight',
            dpi=400, pad_inches=0)


def create_bar_plots(action_distributions, alg_maps, possible_actions, output_dir, fontsize, figsize, facecolor):

    max_step = max(action_distributions[0].keys())
    final_distributions = [distribution[max_step]
                           for distribution in action_distributions]

    df_list = []
    for action in possible_actions:
        action_list = [action.title()]
        for distribution in final_distributions:
            action_list.append(np.mean(distribution[action]))
        df_list.append(action_list)

    df = pd.DataFrame(df_list, columns=[
                      'Probabilities'] + [alg_map["legend"] for alg_map in alg_maps])
    plt.figure(figsize = figsize)
    plt.rcParams.update({'font.size': fontsize})
    fig = df.plot(x='Probabilities', kind='bar', stacked=False,
                  rot=0, grid=True, color=sns.color_palette("husl", len(alg_maps)),
                  fontsize=fontsize, figsize=figsize).get_figure()
    plt.gca().patch.set_facecolor(facecolor)
    plt.grid(color='w', linestyle='solid', alpha=0.5)
    plt.yticks(np.arange(0, 1, 0.1), fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel('Intents', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    fig.savefig(os.path.join(output_dir, "final_action_probabilities.pdf"),
                dpi=400, bbox_inches='tight', pad_inches=0)
