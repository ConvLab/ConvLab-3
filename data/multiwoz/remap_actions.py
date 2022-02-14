from zipfile import ZipFile, ZIP_DEFLATED
from shutil import copy2, rmtree
import json
import os
from tqdm import tqdm


def get_keyword_domains(turn):
    keyword_domains = []
    text = turn['text']
    for d in ["Hotel", "Restaurant", "Train"]:
        if d.lower() in text.lower():
            keyword_domains.append(d)
    return keyword_domains


def get_current_domains_from_act(dialog_acts):

    current_domains_temp = []
    for dom_int in dialog_acts:
        domain, intent = dom_int.split('-')
        if domain in ["general", "Booking"]:
            continue
        if domain not in current_domains_temp:
            current_domains_temp.append(domain)

    return current_domains_temp


def get_next_user_act_domains(ori_dialog, turn_id):
    domains = []
    try:
        next_user_act = ori_dialog['log'][turn_id + 1]['dialog_act']
        domains = get_current_domains_from_act(next_user_act)
    except:
        # will fail if system act is the last act of the dialogue
        pass
    return domains


def check_domain_booked(turn, booked_domains):

    booked_domain_current = None
    for domain in turn['metadata']:
        if turn['metadata'][domain]["book"]["booked"] and domain not in booked_domains:
            booked_domain_current = domain
            booked_domains.append(domain)
    return booked_domains, booked_domain_current


def flatten_acts(dialog_acts):
    flattened_acts = []
    for dom_int in dialog_acts:
        domain, intent = dom_int.split('-')
        for slot_value in dialog_acts[dom_int]:
            slot = slot_value[0]
            value = slot_value[1]
            flattened_acts.append((domain, intent, slot, value))

    return flattened_acts


def deflat_acts(flattened_acts):

    dialog_acts = dict()

    for act in flattened_acts:
        domain, intent, slot, value = act
        if f"{domain}-{intent}" not in dialog_acts.keys():
            dialog_acts[f"{domain}-{intent}"] = [[slot, value]]
        else:
            dialog_acts[f"{domain}-{intent}"].append([slot, value])

    return dialog_acts


def remap_acts(flattened_acts, current_domains, booked_domain=None, keyword_domains_user=None,
               keyword_domains_system=None, current_domain_system=None, next_user_domain=None):

    # We now look for all cases that can happen: Booking domain, Booking within a domain or taxi-inform-car for booking
    error = 0
    remapped_acts = []

    # if there is more than one current domain or none at all, we try to get booked domain differently
    if len(current_domains) != 1 and booked_domain:
        current_domains = [booked_domain]
    elif len(current_domains) != 1 and len(keyword_domains_user) == 1:
        current_domains = keyword_domains_user
    elif len(current_domains) != 1 and len(keyword_domains_system) == 1:
        current_domains = keyword_domains_system
    elif len(current_domains) != 1 and len(current_domain_system) == 1:
        current_domains = current_domain_system
    elif len(current_domains) != 1 and len(next_user_domain) == 1:
        current_domains = next_user_domain

    for act in flattened_acts:
        try:
            domain, intent, slot, value = act
            if f"{domain}-{intent}-{slot}" == "Booking-Book-Ref":
                # We need to remap that booking act now
                assert len(current_domains) == 1, "Can not resolve booking-book act because there are more current domains"
                remapped_acts.append((current_domains[0], "Book", "none", "none"))
                remapped_acts.append((current_domains[0], "Inform", "Ref", value))
            elif domain == "Booking" and intent == "Book" and slot != "Ref":
                # the book intent is here actually an inform intent according to the data
                remapped_acts.append((current_domains[0], "Inform", slot, value))
            elif domain == "Booking" and intent == "Inform":
                # the inform intent is here actually a request intent according to the data
                remapped_acts.append((current_domains[0], "RequestBook", slot, value))
            elif domain == "Booking" and intent in ["NoBook", "Request"]:
                remapped_acts.append((current_domains[0], intent, slot, value))
            elif f"{domain}-{intent}-{slot}" == "Taxi-Inform-Car":
                # taxi-inform-car actually triggers the booking and informs on a car
                remapped_acts.append((domain, "Book", "none", "none"))
                remapped_acts.append((domain, intent, slot, value))
            elif f"{domain}-{intent}-{slot}" in ["Train-Inform-Ref", "Train-OfferBooked-Ref"]:
                # train-inform/offerbooked-ref actually triggers the booking and informs on the reference number
                remapped_acts.append((domain, "Book", "none", "none"))
                remapped_acts.append((domain, "Inform", slot, value))
            elif domain == "Train" and intent == "OfferBook":
                # make offerbook consistent with RequestBook above
                remapped_acts.append(("Train", "RequestBook", slot, value))
            elif domain == "Train" and intent == "OfferBooked" and slot != "Ref":
                # this is actually an inform act
                remapped_acts.append((domain, "Inform", slot, value))
            else:
                remapped_acts.append(act)
        except Exception as e:
            print("Error detected:", e)
            error += 1

    return remapped_acts, error


def preprocess():
    original_data_dir = 'MultiWOZ_2.1'
    new_data_dir = 'data'

    if not os.path.exists(original_data_dir):
        original_data_zip = 'MultiWOZ_2.1.zip'
        if not os.path.exists(original_data_zip):
            raise FileNotFoundError(
                f'cannot find original data {original_data_zip} in multiwoz21/, should manually download MultiWOZ_2.1.zip from https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)
    for filename in os.listdir(original_data_dir):
        if 'db' in filename:
            copy2(f'{original_data_dir}/{filename}', new_data_dir)

    original_data = json.load(open(f'{original_data_dir}/data.json'))
    global init_ontology, cnt_domain_slot

    val_list = set(open(f'{original_data_dir}/valListFile.txt').read().split())
    test_list = set(open(f'{original_data_dir}/testListFile.txt').read().split())

    errors = 0

    for ori_dialog_id, ori_dialog in tqdm(original_data.items()):
        if ori_dialog_id in val_list:
            split = 'validation'
        elif ori_dialog_id in test_list:
            split = 'test'
        else:
            split = 'train'

        # add information to which split the dialogue belongs
        ori_dialog['split'] = split
        current_domains_user = []
        current_domains_system = []
        booked_domains = []

        for turn_id, turn in enumerate(ori_dialog['log']):

            # if it is a user turn, try to extract the current domain
            if turn_id % 2 == 0:
                dialog_acts = turn.get('dialog_act', [])

                keyword_domains_user = get_keyword_domains(turn)
                current_domains_temp = get_current_domains_from_act(dialog_acts)
                current_domains_user = current_domains_temp if current_domains_temp else current_domains_user
            else:

                dialog_acts = turn.get('dialog_act', [])
                if dialog_acts:
                    # only need to go through that process if we have a dialogue act

                    keyword_domains_system = get_keyword_domains(turn)
                    current_domains_temp = get_current_domains_from_act(dialog_acts)
                    current_domains_system = current_domains_temp if current_domains_temp else current_domains_system

                    booked_domains, booked_domain_current = check_domain_booked(turn, booked_domains)
                    next_user_domains = get_next_user_act_domains(ori_dialog, turn_id)

                    flattened_acts = flatten_acts(dialog_acts)
                    remapped_acts, error_local = remap_acts(flattened_acts, current_domains_user,
                                                            booked_domain_current, keyword_domains_user,
                                                            keyword_domains_system, current_domains_system,
                                                            next_user_domains)
                    errors += error_local

                    if error_local > 0:
                        print(ori_dialog_id)

                    deflattened_remapped_acts = deflat_acts(remapped_acts)
                    turn['dialog_act'] = deflattened_remapped_acts

    print("Errors:", errors)
    json.dump(original_data, open(f'{new_data_dir}/data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(original_data_dir)
    rmtree(new_data_dir)


if __name__ == '__main__':
    preprocess()
