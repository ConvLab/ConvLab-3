import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--dir', type=str, default="results",
                        help='Root dir for tensorboard files')
    parser.add_argument('--tb-dir', type=str, default="TB_summary",
                        help='The last dir for tensorboard files')
    parser.add_argument("--map-file", type=str, default="results/map.json")
    parser.add_argument("--out-file", type=str, default="results/fig")
    parser.add_argument("--max-dialogues", type=int, default=0)
    parser.add_argument("--fill-between", type=float, default=0.3,
                        help="the transparency of the std err area")

    args = parser.parse_args()
    return args


def read_data(exp_dir, tb_dir, map_file):
    f_map = json.load(open(map_file))
    data = {}
    for m in f_map:
        data[m["legend"]] = read_dir(exp_dir, tb_dir, m["dir"])
    return data


def read_dir(exp_dir, tb_dir, method_dir):
    dfs = []
    for dir_name in tqdm(glob(os.path.join(exp_dir, method_dir, "*")), ascii=True, desc=method_dir):
        df = read_tb_data(os.path.join(dir_name, tb_dir))
        dfs.append(df)
    return dfs


def read_tb_data(in_path):
    # load log data
    event_data = event_accumulator.EventAccumulator(in_path)
    event_data.Reload()
    keys = event_data.scalars.Keys()
    df = pd.DataFrame(columns=keys[1:])
    for key in keys:
        w_times, step_nums, vals = zip(*event_data.Scalars(key))
        df[key] = vals
        df['steps'] = step_nums
    return df


def plot(data, out_file, plot_type="complete_rate", show_image=False, fill_between=0.3, max_dialogues=0, y_label=''):

    legends = [alg for alg in data]
    clrs = sns.color_palette("husl", len(legends))
    plt.figure(plot_type)

    with sns.axes_style("darkgrid"):
        for i, alg in enumerate(legends):

            max_step = min([len(d[plot_type]) for d in data[alg]])
            if max_dialogues > 0:
                max_length = min([len([s for s in d['steps'] if s <= max_dialogues]) for d in data[alg]])
                max_step = min([max_length, max_step])
            print("max_step: ", max_step)

            value = np.array([d[plot_type][:max_step] for d in data[alg]])
            step = np.array([d['steps'][:max_step] for d in data[alg]][0])
            mean, err = np.mean(value, axis=0), np.std(value, axis=0)
            plt.plot(
                step, mean, c=clrs[i], label=alg)

            plt.fill_between(
                step, mean - err,
                mean + err, alpha=fill_between, facecolor=clrs[i])
        # locs, labels = plt.xticks()
        # plt.xticks(locs, labels)
        #plt.yticks(np.arange(10) / 10)
        #plt.yticks([0.5, 0.6, 0.7])
        plt.xlabel('Training dialogues')
        if len(y_label) > 0:
            plt.ylabel(y_label)
        else:
            plt.ylabel(plot_type)
        plt.legend(fancybox=True, shadow=False, ncol=1, loc='lower left')
        plt.savefig(out_file, bbox_inches='tight')

        if show_image:
            plt.show()


if __name__ == "__main__":
    args = get_args()

    y_label_dict = {"complete_rate": 'Complete rate', "success_rate": 'Success rate', 'turns': 'Average turns',
                    'avg_return': 'Average Return'}

    for plot_type in ["complete_rate", "success_rate", 'turns', 'avg_return']:
        file_name, file_extension = os.path.splitext(args.out_file)
        fig_name = f"{file_name}_{plot_type}{file_extension}"

        data = read_data(exp_dir=args.dir, tb_dir=args.tb_dir,
                         map_file=args.map_file)
        plot(data=data,
             out_file=fig_name,
             plot_type=plot_type,
             fill_between=args.fill_between,
             max_dialogues=args.max_dialogues,
             y_label=y_label_dict[plot_type])
