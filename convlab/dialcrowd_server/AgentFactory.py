'''
AgentFactory.py - Session management between agents and dialogue server.
==========================================================================

@author: Songbo and Neo

'''
from convlab2.dialcrowd_server.Goal import _process_goal
import copy
import json
import numpy as np
import torch
from convlab2.dialcrowd_server.SubjectiveFeedbackManager import SubjectiveFeedbackManager
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util import ContextLogger
from configparser import ConfigParser
import logging
import time
import os
import shutil
logger = ContextLogger.getLogger('')


class AgentFactory(object):

    def __init__(self, configPath, savePath, saveFlag=True, task_file=None):
        self.init_agents()
        self.session2agent = {}
        self.historical_sessions = []
        self.savepath = savePath
        self.saveFlag = saveFlag
        self.number_agents_total = 0
        assert task_file is not None, print("YOU NEED TO PASS A TASK FILE FOR OBJECTIVE SUCCESS.")

        self.filepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #get parent directory
        self.filepath = os.path.dirname(self.filepath) #get grandparent directory
        self.filepath = os.path.join(self.filepath, 'human_trial', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        os.makedirs(self.filepath)
        os.makedirs(os.path.join(self.filepath, 'dialogues'))

        shutil.copy(task_file, self.filepath)

        with open(task_file, 'r') as f:
            self.tasks = []
            for line in f:
                line = line.strip()
                self.tasks.append(json.loads(line))

        # These messages will control sessions for dialogueCrowd. Be careful when you change them, particularly for the fisrt two.
        self.ending_message = "Thanks for your participation. You can now click the Blue Finish Button."
        self.query_taskID_message = "Please now enter the 5 digit task number"
        self.willkommen_message = "Welcome to the dialogue system. How can I help you?"
        self.query_feedback_message = "Got it, thanks. Have you found all the information you were looking for and were all necessary entities booked? Please enter 1 for yes, and 0 for no."
        self.ask_rate_again_message = "Please try again. Have you found all the information you were looking for and were all necessary entities booked? Please enter 1 for yes, and 0 for no."

        configparser = ConfigParser()
        configparser.read(configPath)
        agentPath = (configparser.get("AGENT", "agentPath"))
        agentClass = (configparser.get("AGENT", "agentClass"))
        self.maxTurn = int(configparser.get("AGENT", "maxTurn"))
        self.maxNumberAgent = int(configparser.get("AGENT", "maxNumberAgent"))

        mod = __import__(agentPath, fromlist=[agentClass])
        klass = getattr(mod, agentClass)
        self.template_agent_class = klass
        self.template_agent_instances = klass()
        self.policy = self.template_agent_instances.policy
        self.nlu = copy.deepcopy(self.template_agent_instances.nlu)
        self.nlg = copy.deepcopy(self.template_agent_instances.nlg)
        self.template_agent_instances.policy = None
        self.template_agent_instances.nlu = None
        self.template_agent_instances.nlg = None

        self.subjectiveFeedbackEnabled = (
            configparser.getboolean("SUBJECTIVE", "enabled"))
        self.subjectiveFeedbackManager = None
        self.terminateFlag = False

        # TODO
        # subjectiveFeedbackManager should be independent with subjectiveFeedbackEnabled
        # subjectiveFeedbackManager is used for saving every information
        # subjectiveFeedbackEnabled is used for updating the policy through interacting with real users
        if self.subjectiveFeedbackEnabled:
            self.subjectiveFeedbackManager = SubjectiveFeedbackManager(
                configPath,
                self.policy,
                agent_name=self.template_agent_instances.agent_name)

    def init_agents(self):

        self.agents = {}

    def start_call(self, session_id, user_id=None, task_id=None):
        '''
        Locates an agent to take this call and uses that agents start_call method.

        :param session_id: session_id
        :type session_id: string

        :return: start_call() function of agent id (String)
        '''

        agent_id = None

        print(session_id)

        # 1. make sure session_id is not in use by any agent
        if session_id in list(self.session2agent.keys()):
            agent_id = self.session2agent[session_id]

        # 2. check if there is an inactive agent
        if agent_id is None:
            for a_id in list(self.agents.keys()):
                if self.agents[a_id].session_id is None:
                    agent_id = a_id
                    break

        # 3. otherwise create a new agent for this call
        if agent_id is None:
            agent_id = self.new_agent()
        else:
            logger.info('Agent {} has been reactivated.'.format(agent_id))

        # 4. record that this session is with this agent, and that it existed:
        self.session2agent[session_id] = agent_id
        self.historical_sessions.append(session_id)

        # 5. start the call with this agent:
        self.agents[agent_id].session_id = session_id
        self.agents[agent_id].init_session()
        self.agents[agent_id].agent_saves['session_id'] = session_id
        self.agents[agent_id].agent_saves['agent_id'] = agent_id
        self.agents[agent_id].agent_saves['task_id'] = task_id
        self.agents[agent_id].agent_saves['user_id'] = user_id
        return agent_id

    def continue_call(self, agent_id, user_id, userUtterance, task_id):
        '''
        wrapper for continue_call for the specific Agent() instance identified by agent_id

        :param agent_id: agent id
        :type agent_id: string

        :param userUtterance: user input to dialogue agent
        :type userUtterance: str

        :return: string -- the system's response
        '''

        # If user say "bye", end the dialgue. A user must say "bye" to end the conversation.
        if(str.lower(userUtterance).__contains__("bye")):
            self.agents[agent_id].ENDING_DIALOG = True
            self.agents[agent_id].agent_saves['user_id'] = user_id
            self.agents[agent_id].agent_saves['task_id'] = task_id
            self.agents[agent_id].agent_saves['timestamp'] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.end_call(agent_id)
            self.terminateFlag = True
            return self.ending_message

        #     return self.query_feedback_message

        # # This captures the user subjective feedback. "1" is for achieving the goal. "0" is for not achieving the goal.
        # if self.agents[agent_id].ENDING_DIALOG:
        #     if str.lower(userUtterance) in ["1", "0"]:
        #         self.agents[agent_id].USER_RATED = True
        #         self.agents[agent_id].USER_GOAL_ACHIEVED = (
        #             str.lower(userUtterance) == "1")
        #         self.end_call(agent_id)
        #         return self.ending_message
        #     else:
        #         return self.ask_rate_again_message

        # Get system responses.
        with torch.no_grad():
            prompt_str = self.agents[agent_id].response(userUtterance)
        if(self.agents[agent_id].turn >= self.maxTurn):
            self.agents[agent_id].ENDING_DIALOG = True

        return prompt_str

    def end_call(self, agent_id=None, session_id=None):
        '''
        Can pass session_id or agent_id as we use this in cases
            1) normally ending a dialogue, (via agent_id)
            2) cleaning a hung up call      (via session_id)

        :param agent_id: agent id
        :type agent_id: string

        :param session_id: session_id
        :type session_id: string

        :return: None
        '''

        # 1. find the agent if only given session_id
        if agent_id is None:  # implicitly assume session_id is given then
            agent_id = self.retrieve_agent(session_id)
            if not agent_id:
                return
        logger.info('Ending agents %s call' % agent_id)

        # 2. remove session from active list
        session_id = self.agents[agent_id].session_id
        print("SESSION IDDDDDD: ", session_id)

        del self.session2agent[session_id]
        print("SESSION2AGENT: ", self.session2agent)

        # 3. Train the policy according to the subject feedback from the real user.
        # if self.subjectiveFeedbackEnabled:
        #     training_state = self.agents[agent_id].sys_state_history
        #     training_action = self.agents[agent_id].sys_action_history
        #     training_utterance = self.agents[agent_id].sys_utterance_history
        #     training_reward = None #self.agents[agent_id].retrieve_reward()
        #     training_subjectiveFeedback = self.agents[agent_id].USER_GOAL_ACHIEVED
        #     system_outputs = self.agents[agent_id].sys_output_history
        #     try:
        #         prob_history = self.agents[agent_id].action_prob_history
        #     except:
        #         prob_history = []

        #     task_id = self.agents[agent_id].taskID
        #     self.subjectiveFeedbackManager.add_state_action_lists(
        #         training_utterance, training_state, training_action, training_subjectiveFeedback, training_reward,
        #         task_id, system_outputs, prob_history)
        
        if self.saveFlag:
            user_id = self.agents[agent_id].agent_saves['user_id']
            suffix = str(user_id) + "-" + str(session_id).split("\t")[0] + '.pkl'
            save_path = os.path.join(self.filepath, "dialogues", suffix)

            try:
                task_id = self.agents[agent_id].agent_saves['task_id']
                dialogue = self.agents[agent_id].agent_saves["dialogue_info_fundamental"]
                objective_performance = self.get_objective_performance(task_id, dialogue)
                self.agents[agent_id].agent_saves['performance'] = objective_performance
                print("OBJECTIVE PERFORMANCE:", objective_performance)
            except Exception as e:
                print(f"Could not calculate objective performance: {e}")

            torch.save(self.agents[agent_id].agent_saves, save_path)
        # 4. bye bye, agent : (
        self.kill_agent(agent_id)

    def agent2session(self, agent_id):
        '''
        Gets str describing session_id agent is currently on

        :param agent_id: agent id
        :type agent_id: string

        :return: string -- the session id
        '''
        return self.agents[agent_id].session_id

    def retrieve_agent(self, session_id):
        '''
        Returns str describing agent_id.

        :param session_id: session_id
        :type session_id: string

        :return: string -- the agent id
        '''
        if session_id not in list(self.session2agent.keys()):
            logger.error(
                'Attempted to get an agent for unknown session %s' % session_id)
            return ""
        return self.session2agent[session_id]

    def new_agent(self):
        '''
        Creates a new agent to handle some concurrency.
        Here deepcopy is used to create clean copy rather than referencing,
        leaving it in a clean state to commence a new call.

        :return: string -- the agent id
        '''

        agent_id = 'Smith' + str(self.number_agents_total)
        self.number_agents_total += 1

        # This method has efficiency issue. Loading BERT NLU takes too long which will raise errors for socket.
        # Could also just do a deepcopy of everything here and not setting policy to None in the init
        self.agents[agent_id] = copy.deepcopy(self.template_agent_instances)
        self.agents[agent_id].policy = self.policy
        self.agents[agent_id].nlu = self.nlu
        self.agents[agent_id].nlg = self.nlg

        self.agents[agent_id].dst.init_session()

        if self.subjectiveFeedbackEnabled:
            self.agents[agent_id].policy = self.subjectiveFeedbackManager.getUpdatedPolicy(
            )
        #logger.info('Agent {} has been created.'.format(agent_id))

        if len(self.agents) >= self.maxNumberAgent:
            self.kill_inactive_agent()

        logging.info(
            f"Created new agent {agent_id}. We now have {len(self.agents)} agents in total.")

        return agent_id

    def kill_agent(self, agent_id):
        '''

        :param agent_id: agent id
        :type agent_id: string

        :return: None
        '''

        del self.agents[agent_id]
        torch.cuda.empty_cache()

    def power_down_factory(self):
        '''
        Finalise agents, print the evaluation summary and save the policy we close dialogue server.

        :return: None
        '''

        for agent_id in list(self.agents.keys()):
            logger.info('Summary of agent: %s' % agent_id)
        logger.info('Factory handled these sessions: %s' %
                    self.historical_sessions)

    def kill_inactive_agent(self):
        '''
        Kill inactive agent in the agents list if there are too many agents running.
        '''
        con = 0
        for a_id in list(self.agents.keys()):
            if self.agents[a_id].is_inactive():

                session_id = self.agents[a_id].session_id

                del self.session2agent[session_id]
                self.kill_agent(a_id)
                con += 1
        logger.info('%s of agents are killed.' % con)

    def get_objective_performance(self, task_id, dialogue):

        goal = self.task_id_to_goal(task_id)
        evaluator = MultiWozEvaluator()

        user_acts = []
        system_acts = []
        belief_states = []

        for turn in dialogue:
            system_acts.append(turn['output_action'])
            user_acts.append(turn['state']['user_action'])
            belief_states.append(turn['state']['belief_state'])
        performance_dict = evaluator.evaluate_dialog(goal, user_acts, system_acts, belief_states)

        return performance_dict

    def task_id_to_goal(self, task_id):
        goal = None
        for task in self.tasks:
            if task_id == task['taskID']:
                goal = task
                break
        return _process_goal(goal)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# END OF FILE
