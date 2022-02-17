

class BookingActRemapper:

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_domains_user = []
        self.current_domains_system = []
        self.booked_domains = []

    def retrieve_current_domain_from_user(self, turn_id, ori_dialog):
        prev_user_turn = ori_dialog[turn_id - 1]

        dialog_acts = prev_user_turn.get('dialog_act', [])
        keyword_domains_user = get_keyword_domains(prev_user_turn)
        current_domains_temp = get_current_domains_from_act(dialog_acts)
        self.current_domains_user = current_domains_temp if current_domains_temp else self.current_domains_user
        next_user_domains = get_next_user_act_domains(ori_dialog, turn_id)

        return keyword_domains_user, next_user_domains

    def retrieve_current_domain_from_system(self, turn_id, ori_dialog):

        system_turn = ori_dialog[turn_id]
        dialog_acts = system_turn.get('dialog_act', [])
        keyword_domains_system = get_keyword_domains(system_turn)
        current_domains_temp = get_current_domains_from_act(dialog_acts)
        self.current_domains_system = current_domains_temp if current_domains_temp else self.current_domains_system
        booked_domain_current = self.check_domain_booked(system_turn)

        return keyword_domains_system, booked_domain_current

    def remap(self, turn_id, ori_dialog):

        keyword_domains_user, next_user_domains = self.retrieve_current_domain_from_user(turn_id, ori_dialog)
        keyword_domains_system, booked_domain_current = self.retrieve_current_domain_from_system(turn_id, ori_dialog)

        # only need to remap if there is a dialog action labelled
        dialog_acts = ori_dialog[turn_id].get('dialog_act', [])
        spans = ori_dialog[turn_id].get('span_info', [])
        if dialog_acts:

            flattened_acts = flatten_acts(dialog_acts)
            flattened_spans = flatten_span_acts(spans)
            remapped_acts, error_local = remap_acts(flattened_acts, self.current_domains_user,
                                                    booked_domain_current, keyword_domains_user,
                                                    keyword_domains_system, self.current_domains_system,
                                                    next_user_domains)

            remapped_spans, _ = remap_acts(flattened_spans, self.current_domains_user,
                                               booked_domain_current, keyword_domains_user,
                                               keyword_domains_system, self.current_domains_system,
                                               next_user_domains)

            deflattened_remapped_acts = deflat_acts(remapped_acts)
            deflattened_remapped_spans = deflat_span_acts(remapped_spans)

            return deflattened_remapped_acts, deflattened_remapped_spans
        else:
            return dialog_acts, spans

    def check_domain_booked(self, turn):

        booked_domain_current = None
        for domain in turn['metadata']:
            if turn['metadata'][domain]["book"]["booked"] and domain not in self.booked_domains:
                booked_domain_current = domain.capitalize()
                self.booked_domains.append(domain)
        return booked_domain_current


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
        next_user_act = ori_dialog[turn_id + 1]['dialog_act']
        domains = get_current_domains_from_act(next_user_act)
    except:
        # will fail if system act is the last act of the dialogue
        pass
    return domains


def flatten_acts(dialog_acts):
    flattened_acts = []
    for dom_int in dialog_acts:
        domain, intent = dom_int.split('-')
        for slot_value in dialog_acts[dom_int]:
            slot = slot_value[0]
            value = slot_value[1]
            flattened_acts.append((domain, intent, slot, value))

    return flattened_acts


def flatten_span_acts(span_acts):

    flattened_acts = []
    for span_act in span_acts:
        domain, intent = span_act[0].split("-")
        flattened_acts.append((domain, intent, span_act[1], span_act[2:]))
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


def deflat_span_acts(flattened_acts):

    dialog_span_acts = []
    for act in flattened_acts:
        domain, intent, slot, value = act
        if value == 'none':
            continue
        new_act = [f"{domain}-{intent}", slot]
        new_act.extend(value)
        dialog_span_acts.append(new_act)

    return dialog_span_acts


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
                remapped_acts.append((current_domains[0], "OfferBook", slot, value))
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
            elif domain == "Train" and intent == "OfferBooked" and slot != "Ref":
                # this is actually an inform act
                remapped_acts.append((domain, "Inform", slot, value))
            else:
                remapped_acts.append(act)
        except Exception as e:
            print("Error detected:", e)
            error += 1

    return remapped_acts, error