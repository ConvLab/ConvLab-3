import re, os, time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy


def extract_performance(log_path, ppo=False):
    with open(log_path, 'r') as log_file:

        epoch_list = []
        dialog_counter = []
        complete_rate_list = []
        success_rate_list = []
        total_return_list = []

        for line in log_file:
            result = re.search("root:Evaluating at Epoch: (.*)", line)
            if result is not None:
                epoch = int(result.group(1).split(' ')[0])
                if ppo:
                    dialog_counter.append(epoch * 1000)
                else:
                    dialog_counter.append(epoch * 2)
                epoch_list.append(epoch)
            result = re.search("root:All_user_sim (.*)", line)
            if result is not None:
                complete_rate = float(result.group(1).split(' ')[-1])
                complete_rate_list.append(complete_rate)
            result = re.search("root:All_evaluator (.*)", line)
            if result is not None:
                success_rate = float(result.group(1).split(' ')[-1])
                success_rate_list.append(success_rate)
            result = re.search("root:total_return (.*)", line)
            if result is not None:
                total_return = float(result.group(1).split(' ')[-1])
                total_return_list.append(total_return)

    return epoch_list, complete_rate_list, success_rate_list, dialog_counter, total_return_list


def plot_performance(log_paths, plot_success=True, labels=None, ppo=None):
    best_performance = []
    clrs = sns.color_palette("husl", 5)
    if labels is None:
        labels = log_paths

    with sns.axes_style("darkgrid"):
        for i, path in enumerate(log_paths):
            epoch, complete_rate, success_rate, dialog_counter = extract_performance(path, ppo=ppo[i])
            if plot_success:
                y_label = 'Success Rate'
                performance = success_rate
            else:
                y_label = 'Complete Rate'
                performance = complete_rate

            best_performance.append((path, np.max(performance)))
            plt.plot(dialog_counter[:len(performance) - 1], performance[:-1], c=clrs[i], label=labels[i])

        print("Best Performance:", best_performance)

        plt.xlabel('Dialogs')
        plt.ylabel(y_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.65, 0.35), fancybox=True, shadow=True, ncol=1)
        plt.show()


def plot_performance_across_seeds(log_dir_ensemble_pairs, plot_success=True, labels=None, plot_return=False,
                                  save_path=None):
    best_performance = []
    clrs = sns.color_palette("husl", 6)

    start_performance = [0.055, 0.0725, 0.0825, 0.1, 0.11, 0.125]

    linestyles = ['-', '--', '-.', ':']

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    with sns.axes_style("darkgrid"):
        for i, (direc, ensemble_num) in enumerate(log_dir_ensemble_pairs):
            complete_rate_across_seeds, success_rate_across_seeds, total_return_across_seeds = [], [], []
            for file in os.listdir(direc):
                if ensemble_num is False:
                    epoch, complete_rate, success_rate, dialog_counter, total_return = extract_performance(
                        os.path.join(direc, file + "/log.txt"), ppo=True)
                    complete_rate = complete_rate[:len(dialog_counter)]
                    success_rate = success_rate[:len(dialog_counter)]
                    total_return = total_return[:len(dialog_counter)]
                    complete_rate = np.array(complete_rate)
                    success_rate = np.array(success_rate)
                    total_return = np.array(total_return)
                    complete_rate_across_seeds.append(complete_rate)
                    success_rate_across_seeds.append(success_rate)
                    total_return_across_seeds.append(total_return)
                else:
                    if file.startswith(str(ensemble_num)):
                        epoch, complete_rate, success_rate, dialog_counter, total_return = extract_performance(
                            os.path.join(direc, file + "/log.txt"), ppo=False)
                        complete_rate = complete_rate[:len(dialog_counter)]
                        old_complete_rate = deepcopy(complete_rate)
                        complete_rate = []
                        complete_rate.append(start_performance[ensemble_num - 1])
                        for j in range(1, len(old_complete_rate)):
                            complete_rate.append(old_complete_rate[j - 1])

                        success_rate = success_rate[:len(dialog_counter)]
                        total_return = total_return[:len(dialog_counter)]
                        complete_rate = np.array(complete_rate)
                        success_rate = np.array(success_rate)
                        total_return = np.array(total_return)
                        complete_rate_across_seeds.append(complete_rate)
                        success_rate_across_seeds.append(success_rate)
                        total_return_across_seeds.append(total_return)

            complete_rate_across_seeds = np.array(complete_rate_across_seeds)
            success_rate_across_seeds = np.array(success_rate_across_seeds)
            total_return_across_seeds = np.array(total_return_across_seeds)

            mean_complete, err_complete = np.mean(complete_rate_across_seeds, axis=0), np.std(
                complete_rate_across_seeds, axis=0)
            mean_success, err_success = np.mean(success_rate_across_seeds, axis=0), np.std(success_rate_across_seeds,
                                                                                           axis=0)
            mean_return, err_return = np.mean(total_return_across_seeds, axis=0), np.std(total_return_across_seeds,
                                                                                         axis=0)

            if plot_return:
                y_label = 'Total Return'
                performance = total_return_across_seeds
            elif plot_success:
                y_label = 'Success Rate'
                performance = success_rate_across_seeds
            else:
                y_label = 'Complete rate'
                performance = complete_rate_across_seeds

            best_performance.append((direc + f"{ensemble_num}", np.max(performance)))

            if plot_return:
                plt.plot(dialog_counter, mean_return, c=clrs[i], label=labels[i])
                plt.fill_between(dialog_counter, mean_return - err_return,
                                 mean_return + err_return, alpha=0.3, facecolor=clrs[i])
            elif plot_success:
                plt.plot(dialog_counter, mean_success, c=clrs[i], label=labels[i])
                plt.fill_between(dialog_counter, mean_success - err_success,
                                 mean_success + err_success, alpha=0.3, facecolor=clrs[i])
            else:
                plt.plot(dialog_counter, mean_complete, c=clrs[i], label=labels[i], linestyle=linestyles[i % 4])
                plt.fill_between(dialog_counter, mean_complete - err_complete,
                                 mean_complete + err_complete, alpha=0.3, facecolor=clrs[i])

        print("Best Performance:", best_performance)

        # plt.hlines(y=0.865, xmin=0, xmax=200000, linestyles='--', lw=1)
        # plt.hlines(y=0.57, xmin=0, xmax=200000, linestyles='--', lw=1)

        locs, labels = plt.xticks()
        print(locs)
        labels = ["", "0", "25k", "50k", "75k", "100k", "125k", "150k", "175k", "200k", ""]

        # plt.xticks(locs, [str(epoch)[:-3] + "k" for epoch in locs[1:-1]])
        plt.xticks(locs, labels)
        plt.yticks(np.arange(10) / 10)

        plt.xlabel('Dialogs')
        plt.ylabel(y_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.78, 0.72), fancybox=True, shadow=False, ncol=1)
        plt.savefig(save_path + f"/plot_{current_time}.pdf", bbox_inches='tight')
        plt.show()


def extract_test_performance(log_path):
    with open(log_path, 'r', errors='ignore') as log_file:

        complete_rate_list = []
        return_list = []
        actions = []
        entropy = 0

        for line in log_file:
            result = re.search("Complete: (.*)", line)
            if result is not None:
                complete_rate = float(result.group(1).split(' ')[-1])
                complete_rate_list.append(complete_rate)
            result = re.search("Return: (.*)", line)
            if result is not None:
                return_ = float(result.group(1).split(' ')[-1])
                return_list.append(return_)
            result = re.search("Actions in turn: (.*)", line)
            if result is not None:
                action_number = float(result.group(1).split(' ')[-1])
                actions.append(action_number)
            result = re.search("Entropy: (.*)", line)
            if result is not None:
                entropy = float(result.group(1).split(' ')[-1])

    return complete_rate_list, return_list, actions, entropy


log_path = "/Users/geishaus/repos/convlab-2/log/"
barwidth = 0.9

for file in os.listdir(log_path):
    if not file.endswith('.log'):
        continue
    file_path = os.path.join(log_path, file)
    complete, ret, actions, entropy = extract_test_performance(file_path)

    print("Algorithm:", file)
    print("Complete rate:", np.mean(complete))
    print("Return:", np.mean(ret))
    print("Median:", np.median(ret))
    print("IQR: ", iqr(ret))
    print("Average action in turn:", np.mean(actions))
    print("Entropy:", entropy)
    print('')
