from convlab.dialog_agent import Agent

class E2EAgentWrapper(Agent):
    def __init__(self, e2e_model, name):
        super().__init__(name=name)
        self.model = e2e_model
        self.model.init_session()

    def init_session(self):
        self.model.init_session()

    def response(self, observation):
        return self.model.response(observation)

    def get_in_da(self):
        return None

    def get_out_da(self):
        return None