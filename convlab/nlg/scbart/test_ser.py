import json
from convlab.nlg.evaluate import fine_SER

nlg_output = json.load(open('nlg_output.json', 'r'))

# acts: a list of dialog actions (nlg inputs)
# [  
#   [
#      [
#         "book",
#         "restaurant",
#         "none",
#         "00000038"
#      ],
#      [
#         "recommend",
#         "restaurant",
#         "area",
#         "centre"
#      ],
#      [
#         "request",
#         "restaurant",
#         "price range",
#         "?"
#      ]
#   ]  
# ]
# utts: a list of nlg outputs
# [
#   "I'm sorry, but I have booked you a table in the centre. Your reference number is00000038. Is there a certain price range you prefer?"
# ]
acts, utts = [], []

for entry in nlg_output:
    acts.append(entry['act'])
    utts.append(entry['nlg'])

# missing: number of missed slot/values
# hallucinate: number of hallucinated values
# total: total number of slot
# hallucination_dialogs: a list containing [hallucinated_keyword_1, dialog_acts_1, utterance_excluding_detected_slot/values_1, generated_utterance_1, hallucinated_keword_2, ...]
# missing_dialogs: a list containing [missed_keyword_1, dialog_acts_1, generated_utterance_1, missed_keywords_2, ...]
missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(acts, utts)
