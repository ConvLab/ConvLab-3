from convlab.policy.genTUS.token_map import tokenMap as GenTUSTokenMap


class tokenMap(GenTUSTokenMap):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.default()

    def default(self, only_action=False):
        self.format_tokens = {
            'start_json': '[["',             # 49329
            'sep_token': '", "',            # 1297('",'), 22
            'sep_act': '"], ["',            # 49177
            'end_act': '"]]',            # 42248, 7479, 22
        }

        for token_name in self.format_tokens:
            self.add_token(
                token_name, self.format_tokens[token_name])
