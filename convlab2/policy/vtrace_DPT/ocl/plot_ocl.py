import numpy as np
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('algs', metavar='N', type=str,
                        nargs='+', help='all sub_folders')
    parser.add_argument('--dir_path', default='')
    parser.add_argument('--timeline', default='')

    args = parser.parse_args()
    return args


def load_json(load_path):

    with open(load_path, "r") as f:
        output = json.load(f)
    return output


def read_dir(algorithm_dir_path):

    seed_dir_paths = [f.path for f in os.scandir(algorithm_dir_path) if f.is_dir()]
    seed_dir_names = [f.name for f in os.scandir(algorithm_dir_path) if f.is_dir()]

    metrics_list = []

    for seed_dir_name, seed_dir_path in zip(seed_dir_names, seed_dir_paths):
        if seed_dir_name != "plots":
            metrics = load_json(os.path.join(seed_dir_path, 'logs', 'online_metrics.json'))
            metrics_list.append(metrics)

    return metrics_list


def aggregate_across_seeds(algorithm_dir_path):

    metrics_per_seed = read_dir(algorithm_dir_path)
    metrics_aggregated = metrics_per_seed[0]
    for key in metrics_aggregated:
        for seed_metric in metrics_per_seed[1:]:
            metrics_aggregated[key] = np.concatenate([metrics_aggregated[key], seed_metric[key]])
    for key in metrics_aggregated:
        metrics_aggregated[key] = metrics_aggregated[key].reshape(len(metrics_per_seed), -1)

    return metrics_aggregated


def get_metrics(algorithm_dir_path):
    metrics_aggregated = aggregate_across_seeds(algorithm_dir_path)

    performance_dict = {}
    for key in metrics_aggregated:

        lifetime = [lifetime_progress(output) for output in metrics_aggregated[key]]
        lifetime_mean, lifetime_std = np.mean(lifetime, axis=0), np.std(lifetime, axis=0)
        performance_dict[f"lifetime_{key}"] = {"mean": lifetime_mean, "std": lifetime_std}

        average_per_step = [metric_per_step(output) for output in metrics_aggregated[key]]
        step_mean, step_std = np.mean(average_per_step, axis=0), np.std(average_per_step, axis=0)
        performance_dict[f"local_{key}"] = {"mean": step_mean, "std": step_std}

    return performance_dict


def plot_algorithms(dir_path, alg_names, timeline_path=""):
    clrs = sns.color_palette("husl", 5)
    window_size = 500

    plot_path = os.path.join(dir_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    alg_paths = [os.path.join(dir_path, algorithm_name) for algorithm_name in alg_names]
    performances = [get_metrics(alg_path) for alg_path in alg_paths]

    for key in performances[0].keys():
        max_value = -sys.maxsize
        plt.clf()
        for i, performance in enumerate(performances):
            with sns.axes_style("darkgrid"):

                mean = performance[key]['mean']
                std = performance[key]['std']
                steps = np.arange(0, len(mean)) if 'lifetime' in key \
                    else np.arange(int(window_size / 2), len(mean) + int(window_size / 2))
                plt.plot(steps, mean, c=clrs[i], label=f"{alg_names[i]}")
                plt.fill_between(steps, mean - std, mean + std, alpha=0.3, facecolor=clrs[i])
                plt.xticks(np.arange(0, (1 + int(len(mean)) / 1000) * 1000, 1000))
                plt.xticks(fontsize=7, rotation=0)

                max_value = max_value if np.max(mean) < max_value else np.max(mean)

        if 'local' in key and timeline_path:
            add_timeline(max_value, timeline_path)

        plt.xlabel('Training dialogues')
        plt.ylabel(key)
        plt.title(f"{key}")
        plt.legend(fancybox=True, shadow=False, ncol=1, loc='lower left')
        plt.savefig(plot_path + f'/{key}.pdf', bbox_inches='tight')


def add_timeline(max_value, timeline_path):
    timeline = load_json(timeline_path)['timeline']
    for domain, time in timeline.items():
        plt.axvline(x=time, color='k', linestyle='dashed', linewidth=0.5)
        plt.text(time, max_value, domain, rotation=90, verticalalignment='center')


def metric_per_step(metric, window_size=500):

    kernel = np.ones(window_size) / window_size
    average_per_step = np.convolve(metric, kernel, mode='valid')
    adaptation_rate = average_per_step[1:] - average_per_step[:-1]
    average_adaptation_rate = np.convolve(adaptation_rate, kernel, mode='valid')

    return average_per_step


def lifetime_progress(metric):

    progress = []
    for i in range(1, len(metric) + 1):
        number = metric[i - 1]
        if i == 1:
            mean = number
        else:
            mean = mean + 1/i * (number - mean)
        if i > 50:
            progress.append(mean)
    return progress


if __name__ == '__main__':

    args = arg_parser()
    print("Algorithms compared:", args.algs)

    plot_algorithms(args.dir_path, args.algs, args.timeline)
