# separator
SYS_SPEAK = '[sys_speak]'
USR_SPEAK = '[usr_speak]'
START_OF_PRED = '[start_of_pred]'
END_OF_PRED = '[end_of_pred]'
PAD_TOKEN = '<|pad_token|>'
START_OF_INTENT = '[start_of_intent]'
END_OF_INTENT = '[end_of_intent]'
START_OF_SLOT = ''

SPECIAL_TOKENS = [val for name, val in globals().items() if
                  str.isupper(name) and isinstance(val, str) and val and val[0] == '[' and val[-1] == ']']

assert all(token.islower() for token in SPECIAL_TOKENS)