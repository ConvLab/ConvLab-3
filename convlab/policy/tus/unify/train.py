import json
import os

import torch
import torch.optim as optim
from tqdm import tqdm
from convlab.policy.tus.multiwoz.analysis import Analysis
from convlab.util import load_dataset, load_ontology


def check_device():
    if torch.cuda.is_available():
        print("using GPU")
        return torch.device('cuda')
    else:
        print("using CPU")
        return torch.device('cpu')


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.num_epoch = self.config["num_epoch"]
        self.batch_size = self.config["batch_size"]
        self.device = check_device()
        print(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.config["learning_rate"])

        self.ana = Analysis(config)

    def training(self, train_data, test_data=None):

        self.model = self.model.to(self.device)
        if not os.path.exists(self.config["model_dir"]):
            os.makedirs(self.config["model_dir"])

        save_path = os.path.join(
            self.config["model_dir"], self.config["model_name"])

        # best = [0, 0, 0]
        best = {"loss": 100}
        lowest_loss = 100
        for epoch in range(self.num_epoch):
            print("epoch {}".format(epoch))
            total_loss = self.train_epoch(train_data)
            print("loss: {}".format(total_loss))
            if test_data is not None:
                acc = self.eval(test_data)

            if total_loss < lowest_loss:
                best["loss"] = total_loss
                print(f"save model in {save_path}-loss")
                torch.save(self.model.state_dict(), f"{save_path}-loss")

            for acc_type in acc:
                if acc_type not in best:
                    best[acc_type] = 0
                temp = acc[acc_type]["correct"] / acc[acc_type]["total"]
                if best[acc_type] < temp:
                    best[acc_type] = temp
                    print(f"save model in {save_path}-{acc_type}")
                    torch.save(self.model.state_dict(),
                               f"{save_path}-{acc_type}")
            if epoch < 10 and epoch > 5:
                print(f"save model in {save_path}-{epoch}")
                torch.save(self.model.state_dict(),
                           f"{save_path}-{epoch}")
            print(f"save latest model in {save_path}")
            torch.save(self.model.state_dict(), save_path)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        result = {}
        # result = {"id": {"slot": {"prediction": [],"label": []}}}
        count = 0
        for i, data in enumerate(tqdm(data_loader, ascii=True, desc="Training"), 0):
            input_feature = data["input"].to(self.device)
            mask = data["mask"].to(self.device)
            label = data["label"].to(self.device)
            if self.config.get("domain_traget", True):
                domain = data["domain"].to(self.device)
            else:
                domain = None
            self.optimizer.zero_grad()

            loss, output = self.model(input_feature, mask, label, domain)

            loss.backward()
            self.optimizer.step()
            total_loss += float(loss)
            count += 1

        return total_loss / count

    def eval(self, test_data):
        self.model.zero_grad()
        self.model.eval()

        result = {}

        with torch.no_grad():
            correct, total, non_zero_correct, non_zero_total = 0, 0, 0, 0
            for i, data in enumerate(tqdm(test_data, ascii=True, desc="Evaluation"), 0):
                input_feature = data["input"].to(self.device)
                mask = data["mask"].to(self.device)
                label = data["label"].to(self.device)
                output = self.model(input_feature, mask)
                r = parse_result(output, label)
                for r_type in r:
                    if r_type not in result:
                        result[r_type] = {"correct": 0, "total": 0}
                    for n in result[r_type]:
                        result[r_type][n] += float(r[r_type][n])

        for r_type in result:
            temp = result[r_type]['correct'] / result[r_type]['total']
            print(f"{r_type}: {temp}")

        return result


def parse_result(prediction, label):
    # result = {"id": {"slot": {"prediction": [],"label": []}}}
    # dialog_index = ["dialog-id"_"slot-name", "dialog-id"_"slot-name", ...]
    # prdiction = [0, 1, 0, ...] # after max

    _, arg_prediction = torch.max(prediction.data, -1)
    batch_size, token_num = label.shape
    result = {
        "non-zero": {"correct": 0, "total": 0},
        "total": {"correct": 0, "total": 0},
        "turn": {"correct": 0, "total": 0}
    }

    for batch_num in range(batch_size):
        turn_acc = True
        for element in range(token_num):
            result["total"]["total"] += 1
            if label[batch_num][element] > 0:
                result["non-zero"]["total"] += 1

            if arg_prediction[batch_num][element + 1] == label[batch_num][element]:
                if label[batch_num][element] > 0:
                    result["non-zero"]["correct"] += 1
                result["total"]["correct"] += 1

            elif arg_prediction[batch_num][element + 1] == 0 and label[batch_num][element] < 0:
                result["total"]["correct"] += 1

            else:
                if label[batch_num][element] >= 0:
                    turn_acc = False

        result["turn"]["total"] += 1
        if turn_acc:
            result["turn"]["correct"] += 1

    return result


def f1(target, result):
    target_len = 0
    result_len = 0
    tp = 0
    for t, r in zip(target, result):
        if t:
            target_len += 1
        if r:
            result_len += 1
        if r == t and t:
            tp += 1
    precision = 0
    recall = 0
    if result_len:
        precision = tp / result_len
    if target_len:
        recall = tp / target_len
    f1_score = 2 / (1 / precision + 1 / recall)
    return f1_score, precision, recall


def save_data(data, file_name, file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f_name = os.path.join(file_dir, file_name)
    torch.save(data, f_name)
    # with open(f_name, 'wb') as data_obj:
    #     pickle.dump(data, data_obj, pickle.HIGHEST_PROTOCOL)
    print(f"save data to {f_name}")


if __name__ == "__main__":
    import argparse
    import os
    from convlab.policy.tus.multiwoz.transformer import \
        TransformerActionPrediction
    from convlab.policy.tus.unify.usermanager import \
        TUSDataManager
    from torch.utils.data import DataLoader
    from convlab.policy.tus.unify.util import update_config_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--user-config", type=str,
                        default="convlab/policy/tus/unify/exp/default.json")
    parser.add_argument("--force-read-data", '-f', action='store_true',
                        help="Force to read data from scratch")
    parser.add_argument("--dataset", type=str, default="multiwoz21")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)

    args = parser.parse_args()
    config_file = open(args.user_config)
    config = json.load(config_file)
    config_file.close()
    if args.dataset == "all":
        print("merge all datasets...")
        all_dataset = ["multiwoz21", "sgd", "tm1", "tm2", "tm3"]
        datasets = {}
        for dataset in all_dataset:
            datasets[dataset] = load_dataset(dataset,
                                             dial_ids_order=args.dial_ids_order,
                                             split2ratio={'train': args.split2ratio})
        # merge dataset
        raw_data = {}
        for data_type in ["train", "test"]:
            raw_data[data_type] = []
            for dataset in all_dataset:
                raw_data[data_type] += datasets[dataset][data_type]

    elif args.dataset == "sgd+tm":
        print("merge multiple datasets...")
        all_dataset = ["sgd", "tm1", "tm2", "tm3"]
        datasets = {}
        for dataset in all_dataset:
            datasets[dataset] = load_dataset(dataset,
                                             dial_ids_order=args.dial_ids_order,
                                             split2ratio={'train': args.split2ratio})
        # merge dataset
        raw_data = {}
        for data_type in ["train", "test"]:
            raw_data[data_type] = []
            for dataset in all_dataset:
                raw_data[data_type] += datasets[dataset][data_type]

    else:
        print(f"load single dataset {args.dataset}/{args.split2ratio}")
        raw_data = load_dataset(args.dataset,
                                dial_ids_order=args.dial_ids_order,
                                split2ratio={'train': args.split2ratio})

    batch_size = config["batch_size"]

    # load data with "load_data"

    # check train/test data
    data = {"train": {}, "test": {}}
    for data_type in data:
        data[data_type]["data"] = TUSDataManager(
            config, raw_data[data_type])

    # check the embed_dim and update it
    embed_dim = data["train"]["data"].features["input"].shape[-1]
    if embed_dim != config["embed_dim"]:
        config["embed_dim"] = embed_dim
        update_config_file(file_name=args.user_config,
                           attribute="embed_dim", value=embed_dim)

    train_data = DataLoader(data["train"]["data"],
                            batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data["test"]["data"],
                           batch_size=batch_size, shuffle=True)

    model = TransformerActionPrediction(config)

    if "pretrain" in config:
        pretrain_weight = os.path.join(
            f'{config["pretrain"]}_{args.dial_ids_order}', f"model-loss")
        print(f"fine tune based on {pretrain_weight}...")
        model.load_state_dict(torch.load(
            pretrain_weight, map_location=check_device()))
    trainer = Trainer(model, config)
    trainer.training(train_data, test_data)
