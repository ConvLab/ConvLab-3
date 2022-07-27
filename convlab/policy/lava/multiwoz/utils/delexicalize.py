import pickle
import re, pdb, os

import simplejson as json

# temp_path = os.path.dirname(os.path.abspath(__file__))
# fin = open(os.path.join(temp_path, 'mapping.pair'))
# replacements = []
# for line in fin.readlines():
    # tok_from, tok_to = line.replace('\n', '').split('\t')
#     replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

# def insertSpace(token, text):
#     sidx = 0
    # while True:
        # sidx = text.find(token, sidx)
        # if sidx == -1:
            # break
        # if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                # re.match('[0-9]', text[sidx + 1]):
            # sidx += 1
            # continue
        # if text[sidx - 1] != ' ':
            # text = text[:sidx] + ' ' + text[sidx:]
            # sidx += 1
        # if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            # text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        # sidx += 1
#     return text

def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    #TODO pay attention to the slot token, value_time or time
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)
    #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

# digitpat = re.compile('\d+')
# timepat = re.compile("\d{1,2}[:]\d{1,2}")
# pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")
# pricepat = re.compile("\d{1,3}[.]\d{1,2}")

# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..

def prepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction',
               'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    # read databases
    for domain in domains:
        try:
            fin = open('db/' + domain + '_db.json')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append(
                            (normalize(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append(
                            (normalize(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append(
                                (normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append(
                            (normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append(
                            (normalize(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append(
                            (normalize(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append(
                            (normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append(
                            (normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append(
                            (normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    else:
                        pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append(
                (normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('Hills Road'),
                        '[' + domain + '_' + 'address' + ']'))
            dic.append(
                (normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Addenbrookes Hospital'),
                        '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append(
                (normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append(
                (normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Parkside Police Station'),
                        '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    # fin = file('db/' + 'train' + '_db.json')
    fin = open('db/' + 'train' + '_db.json')
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure' or key == 'destination':
                dic.append(
                    (normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic

def AuGPTprepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction',
               'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []


    # read databases
    for domain in domains:
        try:
            fin = open('augpt_mw21_db/db/' + domain + '_db.json')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append(
                            (normalize(val), '[address]'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append(
                                (normalize(val), '[address]'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append(
                                (normalize(val), '[address]'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append(
                                (normalize(val), '[address]'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append(
                                (normalize(val), '[address]'))
                    elif key == 'name':
                        dic.append(
                            (normalize(val), '[name]'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append(
                                (normalize(val), '[name]'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append(
                                (normalize(val), '[name]'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append(
                                (normalize(val), '[name]'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append(
                                (normalize(val), '[name]'))
                    elif key == 'postcode':
                        dic.append(
                            (normalize(val), '[postcode]'))
                    elif key == 'phone':
                        dic.append((val, '[phone]'))
                    elif key == 'trainID':
                        dic.append(
                            (normalize(val), '[id]'))
                    elif key == 'department':
                        dic.append(
                            (normalize(val), '[department]'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append(
                            (normalize(val), '[area]'))
                    elif key == 'food':
                        dic_food.append(
                            (normalize(val), '[food]'))
                    elif key == 'pricerange':
                        dic_price.append(
                            (normalize(val), '[price range]'))
                    else:
                        pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append(
                (normalize('Hills Rd'), '[address]'))
            dic.append((normalize('Hills Road'),
                        '[address]'))
            dic.append(
                (normalize('CB20QQ'), '[postcode]'))
            dic.append(('01223245151', '[phone]'))
            dic.append(('1223245151', '[phone]'))
            dic.append(('0122324515', '[phone]'))
            dic.append((normalize('Addenbrookes Hospital'),
                        '[name]'))

        elif domain == 'police':
            dic.append(
                (normalize('Parkside'), '[address]'))
            dic.append(
                (normalize('CB11JG'), '[postcode]'))
            dic.append(('01223358966', '[phone]'))
            dic.append(('1223358966', '[phone]'))
            dic.append((normalize('Parkside Police Station'),
                        '[name]'))

    # add at the end places from trains
    # fin = file('db/' + 'train' + '_db.json')
    fin = open('augpt_mw21_db/db/' + 'train' + '_db.json')
    db_json = json.load(fin)
    fin.close()

    # for ent in db_json:
        # for key, val in ent.items():
            # if key == 'departure' or key == 'destination':
                # dic.append(
                    # (normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # # add specific values:
    # for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        # dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt

def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?
    return utt


if __name__ == '__main__':
    dic = AuGPTprepareSlotValuesIndependent()
    pickle.dump(dic, open('data/augpt_svdic.pkl', 'wb+'))
