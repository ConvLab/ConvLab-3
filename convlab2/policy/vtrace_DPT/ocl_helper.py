def log_used_budget(start_budget, budget):
    used_budget = []

    for start, remaining in zip(start_budget, budget):
        used = start[1] - remaining[1]
        if used > 0:
            used_budget.append([start[0], used])

    logging.info(f"Used budget: {used_budget}")


def load_budget(load_path):

    with open(load_path, "r") as f:
        budget_info = json.load(f)
        timeline = budget_info['timeline']
        budget = budget_info['budget']
        timeline['end'] = sum([pair[1] for pair in budget])
        name = args.budget_path.split("/")[-1].split(".")[0]
        save_budget_plot(budget, os.path.join(os.path.dirname(args.budget_path), f"{name}.pdf"))
    return timeline, budget


def check_setup(timeline, budget):
    # We need to check whether there is enough budget for the timeline used.
    # For instance, if we only have 800 restaurants but only introduce the next domain after 1000 dialogues according
    # to the timeline, this will not work
    timeline_ = list(timeline.items())
    timeline_.sort(key=lambda x: x[1])
    allowed_domains = []

    for i, domain_start in enumerate(timeline_[:-1]):
        domain = domain_start[0]
        allowed_domains.append(domain)
        necessary_budget = timeline_[i+1][1]
        real_budget = 0
        for pair in budget:
            domain_combi = set(pair[0].split("-"))
            if domain_combi.issubset(set(allowed_domains)):
                real_budget += pair[1]
        assert necessary_budget <= real_budget, "The necessary budget is higher than the real budget!"


def save_budget_plot(budget, save_path):
    import matplotlib.pyplot as plt
    x = [pair[0] for pair in budget]
    y = [pair[1] for pair in budget]
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Domain combinations")
    plt.ylabel("Budget per combination")
    plt.title("Budget for every domain combination")
    plt.xticks(x_pos, x, rotation='vertical')
    plt.tight_layout()
    plt.savefig(save_path)


def update_online_metrics(online_metrics, metrics, log_save_path, tb_writer):
    for metric in metrics:
        for key in online_metrics:
            online_metrics[key].append(metric[key])
            tb_writer.add_scalar(f"lifetime_{key}", np.mean(online_metrics[key]), len(online_metrics[key]))
    json.dump(online_metrics, open(os.path.join(log_save_path, 'online_metrics.json'), 'w'))


def sample_domains(allowed_domains, budget):

    distribution = []
    for pair in budget:
        domain_combi = set(pair[0].split("-"))
        if domain_combi.issubset(set(allowed_domains)):
            distribution.append(pair[1])
        else:
            distribution.append(0)

    distribution = np.array(distribution, dtype=np.float32)
    distribution /= sum(distribution)
    domain_idx = np.random.choice(np.arange(0, len(budget)), p=distribution)
    budget[domain_idx][1] -= 1
    return budget[domain_idx][0], budget