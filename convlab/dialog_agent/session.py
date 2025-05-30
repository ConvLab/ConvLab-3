"""Dialog controller classes."""
from abc import ABC, abstractmethod
import random
from convlab.dialog_agent.agent import Agent


class Session(ABC):

    """Base dialog session controller, which manages the agents to conduct a complete dialog session.
    """

    @abstractmethod
    def next_agent(self):
        """Decide the next agent to generate a response.

        In this base class, this function returns the index randomly.

        Returns:
            next_agent (Agent): The index of the next agent.
        """
        pass

    @abstractmethod
    def next_response(self, observation):
        """Generated the next response.

        Args:
            observation (str or dict): The agent observation of next agent.
        Returns:
            response (str or dict): The agent's response.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the agent variables for a new session."""
        pass


class BiSession(Session):
    """The dialog controller which aggregates several agents to conduct a complete dialog session.

    Attributes:
        sys_agent (Agent):
            system dialog agent.

        user_agent (Agent):
            user dialog agent.

        kb_query (KBquery):
            knowledge base query tool.

        dialog_history (list):
            The dialog history, formatted as [[user_uttr1, sys_uttr1], [user_uttr2, sys_uttr2], ...]
    """

    def __init__(self, sys_agent: Agent, user_agent: Agent, kb_query=None, evaluator=None):
        """
        Args:
            sys_agent (Agent):
                An instance of system agent.

            user_agent (Agent):
                An instance of user agent.

            kb_query (KBquery):
                An instance of database query tool.

            evaluator (Evaluator):
                An instance of evaluator.
        """
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.kb_query = kb_query
        self.evaluator = evaluator

        self.dialog_history = []
        self.__turn_indicator = 0

        # self.init_session()

    def next_agent(self):
        """The user and system agent response in turn."""
        if self.__turn_indicator % 2 == 0:
            next_agent = self.user_agent
            agent = "user"
        else:
            next_agent = self.sys_agent
            agent = "sys"
        self.__turn_indicator += 1
        # print(agent + " " + str(self.__turn_indicator))
        return next_agent

    def next_response(self, observation, **kwargs):
        next_agent = self.next_agent()
        return next_agent.response(observation, **kwargs)

    def _system_action(self):
        if self.user_agent.nlu is not None:
            # predict user action by user nlu
            return None
        return self.sys_agent.output_action

    def next_turn(self, last_observation):
        """Conduct a new turn of dialog, which consists of the system response and user response.

        The variable type of responses can be either 1) str or 2) dialog act, depends on the dialog mode settings of the
        two agents which are supposed to be the same.

        Args:
            last_observation:
                Last agent response.
        Returns:
            sys_response:
                The response of system.

            user_response:
                The response of user simulator.

            session_over (boolean):
                True if session ends, else session continues.

            reward (float):
                The reward given by the user.
        """
        user_response_type = self.user_agent.get_response_type()
        if user_response_type == "utterance_to_user":
            if type(last_observation) is not str:
                raise Exception(
                    "last_observation from the system should be string")
            system_action = self._system_action()
            user_response = self.next_response(
                last_observation, action=system_action)
        elif user_response_type == 'action_utterance_to_user':
            if len(last_observation) == 0:
                last_observation = ('', [])
            system_response, system_action = last_observation
            user_response = self.next_response(
                system_response, action=system_action)
        elif user_response_type == "need_conduct_user":
            sys_conduct = self.sys_agent.policy.get_conduct()
            user_response = self.next_response(
                last_observation, conduct=sys_conduct)
        else:
            user_response = self.next_response(last_observation)

        if self.evaluator:
            if hasattr(self.sys_agent, 'dst'):
                self.evaluator.add_sys_da(
                    self.user_agent.get_in_da(), self.sys_agent.dst.state['belief_state'])
            else:
                self.evaluator.add_sys_da(self.user_agent.get_in_da())
            self.evaluator.add_usr_da(self.user_agent.get_out_da())

        session_over = self.user_agent.is_terminated()
        if hasattr(self.sys_agent, 'dst'):
            self.sys_agent.dst.state['terminated'] = session_over

        # Get user emotion
        self.update_user_emotion()

        # if session_over and self.evaluator:
        # prec, rec, f1 = self.evaluator.inform_F1()
        # print('inform prec. {} rec. {} F1 {}'.format(prec, rec, f1))
        # print('book rate {}'.format(self.evaluator.book_rate()))
        # print('task success {}'.format(self.evaluator.task_success()))
        reward = self.user_agent.get_reward(
        ) if self.evaluator is None else self.evaluator.get_reward(session_over)
        sys_response = self.next_response(user_response)
        self.dialog_history.append([self.user_agent.name, user_response])
        self.dialog_history.append([self.sys_agent.name, sys_response])
        return sys_response, user_response, session_over, reward

    def update_user_emotion(self):
        if hasattr(self.sys_agent, 'dst'):
            if not hasattr(self.sys_agent.dst, 'get_emotion') and hasattr(self.user_agent.policy, 'get_emotion'):
                emotion = self.user_agent.policy.get_emotion().lower()
                self.sys_agent.dst.state['user_emotion'] = emotion

    def next_turn_two_way(self, system_utterance, system_action):
        """Conduct a new turn of dialog, which consists of the system response and user response.

        The variable type of responses can be either 1) str or 2) dialog act, depends on the dialog mode settings of the
        two agents which are supposed to be the same.

        Args:
            system_utterance:
                The utterance of system.

            system_action:
                The action of system.

        Returns:
            sys_response:
                The response of system.

            user_response:
                The response of user simulator.

            session_over (boolean):
                True if session ends, else session continues.

            reward (float):
                The reward given by the user.
        """
        user = self.next_agent()
        user_response = user.response(system_utterance, system_action)

        if self.evaluator:
            self.evaluator.add_sys_da(
                self.user_agent.get_in_da(), self.sys_agent.dst.state['belief_state'])
            self.evaluator.add_usr_da(self.user_agent.get_out_da())

        session_over = self.user_agent.is_terminated()
        if hasattr(self.sys_agent, 'dst'):
            self.sys_agent.dst.state['terminated'] = session_over
        # if session_over and self.evaluator:
            # prec, rec, f1 = self.evaluator.inform_F1()
            # print('inform prec. {} rec. {} F1 {}'.format(prec, rec, f1))
            # print('book rate {}'.format(self.evaluator.book_rate()))
            # print('task success {}'.format(self.evaluator.task_success()))
        reward = self.user_agent.get_reward(
        ) if self.evaluator is None else self.evaluator.get_reward()

        system = self.next_agent()
        sys_response = system.response(user_response)
        sys_action = system.get_out_da()
        self.dialog_history.append([self.user_agent.name, user_response])
        self.dialog_history.append([self.sys_agent.name, sys_response])
        return sys_response, sys_action, user_response, session_over, reward

    def train_policy(self):
        """
        Train the parameters of system agent policy.
        """
        self.sys_agent.policy.train()

    def init_session(self, **kwargs):
        self.sys_agent.init_session()
        self.user_agent.init_session(**kwargs)
        if self.evaluator:
            self.evaluator.add_goal(self.user_agent.policy.get_goal())
        self.dialog_history = []
        self.__turn_indicator = 0


class DealornotSession(Session):
    """A special session for Deal or Not dataset, which is a object dividing negotiation task."""

    def __init__(self, alice, bob):
        self.alice = alice
        self.bob = bob
        self.__turn_indicator = 0
        self.init_session()
        self.current_agent = None
        self.dialog_history = []

    def next_agent(self):
        """Alice and Bob agents response in turn."""
        if self.__turn_indicator % 2 == 0:
            next_agent = self.alice
        else:
            next_agent = self.bob
        self.__turn_indicator += 1
        return next_agent

    def next_response(self, observation):
        agent = self.next_agent()
        self.current_agent = agent
        model_response = self.current_agent.response(observation)
        self.dialog_history.append(model_response)
        return model_response

    def is_terminated(self):
        if self.current_agent.is_terminated():
            return True

    def get_rewards(self, ctxs):
        """Return the rewards of alice and bob.

        Returns:
            reward_1 (float):
                Reward of Alice.

            reward_2 (float):
                Reward of Bob.
        """
        choices = []
        for agent in [self.alice, self.bob]:
            choice = agent.choose()
            choices.append(choice)

        agree, rewards = self.alice.domain.score_choices(choices, ctxs)
        return agree, rewards

    def init_session(self):
        self.__turn_indicator = random.choice([0, 1])
        self.alice.init_session()
        self.bob.init_session()
        self.current_agent = None
        self.dialog_history = []
