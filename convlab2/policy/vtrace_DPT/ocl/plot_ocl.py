import numpy as np
import json
import matplotlib.pyplot as plt


def load_json(load_path):

    with open(load_path, "r") as f:
        output = json.load(f)
    return output


def metric_per_step(metric, window_size=500):

    kernel = np.ones(window_size) / window_size
    average_per_step = np.convolve(metric, kernel, mode='valid')
    adaptation_rate = average_per_step[1:] - average_per_step[:-1]
    average_adaptation_rate = np.convolve(adaptation_rate, kernel, mode='valid')

    return average_per_step, average_adaptation_rate


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

    path = "ocl/ddpt_0.2/logs/online_metrics.json"
    path_2 = "ocl/ddpt_0.8/logs/online_metrics.json"
    metrics = load_json(path)
    metrics_2 = load_json(path_2)
    window_size = 500

    success_per_step, adaptation_success = metric_per_step(metrics['success'], window_size)
    return_per_step, adaptation_return = metric_per_step(metrics['return'], window_size)
    lifetime = lifetime_progress(metrics['success'])
    return_per_step_2, adaptation_return_2 = metric_per_step(metrics_2['return'], window_size)
    success_per_step_2, adaptation_success_2 = metric_per_step(metrics_2['success'], window_size)
    lifetime_2 = lifetime_progress(metrics_2['success'])

    plt.plot(np.arange(0, len(lifetime)), lifetime, label="DDPT_0.2")
    plt.plot(np.arange(0, len(lifetime_2)), lifetime_2, label="DDPT_0.8")
    plt.xlabel('Average lifetime success after t dialogues')
    plt.ylabel("Success")
    plt.title("Average lifetime success")
    plt.legend()
    plt.savefig("ocl/average_lifetime_success.pdf")
    plt.show()

    plt.plot(np.arange(int(window_size / 2), len(success_per_step) + int(window_size / 2)), success_per_step, label="DDPT_0.2")
    plt.plot(np.arange(int(window_size / 2), len(success_per_step_2) + int(window_size / 2)), success_per_step_2, label="DDPT_0.8")
    plt.xlabel('Training dialogues')
    plt.ylabel("Success")
    plt.title("Average success per step")
    plt.legend()
    plt.savefig("ocl/average_success.pdf")
    plt.show()

    plt.plot(np.arange(0, len(adaptation_success)), adaptation_success, label=f"adaptation_success")
    plt.plot(np.arange(0, len(adaptation_success_2)), adaptation_success_2, label=f"adaptation_success")
    plt.hlines(0.0, xmin=0, xmax=len(adaptation_success))
    plt.xlabel('Training dialogues')
    plt.ylabel("Adaptation rate")
    plt.title("Average adaptation per step")
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, len(return_per_step)), return_per_step, label=f"success")
    plt.show()
    plt.plot(np.arange(0, len(adaptation_return)), adaptation_return, label=f"adaptation_success")
    plt.hlines(0.0, xmin=0, xmax=len(adaptation_return))
    plt.show()

