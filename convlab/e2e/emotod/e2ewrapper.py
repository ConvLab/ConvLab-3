from convlab.dialog_agent import Agent

class E2EAgentWrapper(Agent):
    def __init__(self, e2e_model, name):
        super().__init__(name=name)
        self.policy = e2e_model
        self.policy.init_session()

    def init_session(self):
        self.policy.init_session()

    def response(self, observation):
        return self.policy.response(observation)

    def get_in_da(self):
        return None

    def get_out_da(self):
        return None
    
    def state_return(self):
        return None