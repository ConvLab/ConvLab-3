import json


class tokenMap:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.token_name = {}
        self.hash_map = {}
        self.debug = False
        self.default()

    def default(self, only_action=False):
        self.format_tokens = {
            'start_json': '{"',
            'start_sentiment': 'sentiment": "',
            'start_emotion': 'emotion": "',
            'start_act': 'action": [["',
            'sep_token': '", "',
            'sep_act': '"], ["',
            'end_act': '"]], "',
            'start_text': 'text": "',
            'end_json': '}',
            'end_json_2': '"}',
            'book': 'book'
        }

        if only_action:
            self.format_tokens['end_act'] = '"]]}'
        for token_name in self.format_tokens:
            self.add_token(
                token_name, self.format_tokens[token_name])

    def add_token(self, token_name, value):
        if token_name in self.token_name and self.debug:
            print(f"---> duplicate token: {token_name}({value})!!!!!!!")

        token_id = self.tokenizer(str(value), add_special_tokens=False)[
            "input_ids"]
        self.token_name[token_name] = {"value": value, "token_id": token_id}
        # print(token_id)
        hash_id = token_id[0]
        if hash_id in self.hash_map and self.debug:
            print(
                f"---> conflict hash number {hash_id}: {self.hash_map[hash_id]['name']} and {token_name}")
        self.hash_map[hash_id] = {
            "name": token_name, "value": value, "token_id": token_id}

    def get_info(self, hash_id):
        return self.hash_map[hash_id]

    def get_id(self, token_name):
        # workaround
        # if token_name not in self.token_name[token_name]:
        #     self.add_token(token_name, token_name)
        return self.token_name[token_name]["token_id"]

    def get_token_value(self, token_name):
        return self.token_name[token_name]["value"]

    def token_name_is_in(self, token_name):
        if token_name in self.token_name:
            return True
        return False

    def hash_id_is_in(self, hash_id):
        if hash_id in self.hash_map:
            return True
        return False
