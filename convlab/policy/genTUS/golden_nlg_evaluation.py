import json
import os
import sys
from argparse import ArgumentParser

from convlab.nlg.evaluate import fine_SER
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.util.unified_datasets_util import load_ontology


from tqdm import tqdm
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, help="the testing input file",
                        default="")
    return parser.parse_args()


def ser_v2(actions, utterances, ontology="multiwoz21"):
    ontology = load_ontology(ontology)
    # ERROR Rate
    # get all values in ontology
    val2ds_dict = {}
    for domain_name in ontology['domains']:
        domain = ontology['domains'][domain_name]
        for slot_name in domain['slots']:
            slot = domain['slots'][slot_name]
            if 'possible_values' not in slot:
                continue
            possible_vals = slot['possible_values']
            if len(possible_vals) > 0:
                for val in possible_vals:
                    val2ds_dict[val] = f'{domain_name}-{slot_name}'
    score_list = {"missing": [], "redundant": [], "all": []}
    for da, utterance in zip(actions, utterances):
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        all_domains = set()
        # missing values
        for intent, domain, slot, value in da:
            all_domains.add(domain)
            if intent != "inform":
                continue
            if value in ["yes", "no", "none", "dontcare"]:
                continue
            all_values.add(value.strip().lower())
            if value.strip().lower() not in utterance.lower():
                missing_count += 1
                # print(f"missing: {intent, domain, slot, value} | {utterance}")
            all_count += 1
        if all_count == 0:
            continue

        # redundant values

        for val in val2ds_dict:
            if f' {val.strip().lower()} ' in f' {utterance.strip().lower()} ' and val.strip().lower() not in all_values:
                wlist = val2ds_dict[val].split('-')
                domain, slot = wlist[0], wlist[1]
                if val in ["yes", "no", "none", "dontcare"]:
                    continue
                if domain not in all_domains:
                    continue
                if f' {slot.strip().lower()}' in f' {utterance.strip().lower()} ':
                    if val == "free":  # workaround for "free/hotel-parking" -> yes/hotel-parking
                        continue
                    redundant_count += 1
                    # print("------------------")
                    # print(da)
                    # print(all_values)
                    # print(f"redundant: {val}/{val2ds_dict[val]} | {utterance}")
                    # logger.log(f"redundant: {val}/{val2ds_dict[val]} | {item['prediction']} | {item['utterance']}")
        # logger.log(f"redundant: {redundant_count} | missing_count: {missing_count} |all_count: {all_count}")
        score_list["missing"].append(missing_count/all_count)
        score_list["redundant"].append(redundant_count/all_count)
        score_list["all"].append(all_count)
    ser = {}
    for metric, s in score_list.items():
        ser[metric] = np.mean(s)
    ser["ser"] = (ser["missing"] + ser["redundant"])/ser["all"]
    return ser


def calculate(file_name):
    nlg = TemplateNLG(is_user=True)
    in_file = json.load(open(file_name))
    r = {
        "input": [],
        "golden_acts": [],
        "golden_utts": [],
        "generate_utts": [],
    }
    for dialog in tqdm(in_file['dialog']):
        inputs = dialog["in"]
        labels = json.loads(dialog["out"])
        r["input"].append(inputs)
        r["golden_acts"].append(labels["action"])
        r["golden_utts"].append(labels["text"])
        r["generate_utts"].append(nlg.generate(labels["action"]))

    missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
        r["golden_acts"], r["golden_utts"])
    print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
        "human", missing, total, hallucinate, (missing+hallucinate)/total))
    missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
        r["golden_acts"], r["generate_utts"])
    print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
        "template", missing, total, hallucinate, (missing+hallucinate)/total))
    ontology = "multiwoz21"
    print("SER v2: ",
          ser_v2(r["golden_acts"], r["golden_utts"], ontology), " | ",
          # ser_v2(r["golden_acts"], r["generate_utts"], ontology)
          )
    return r


def main():
    args = arg_parser()
    calculate(args.input_file)


if __name__ == '__main__':
    main()
