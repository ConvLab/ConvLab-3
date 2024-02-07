'''
DialogueServer.py - Web interface for DialCrowd.
==========================================================================

@author: Songbo

'''

from configparser import ConfigParser
from convlab.dialcrowd_server.AgentFactory import AgentFactory
from datetime import datetime
import json
import http.server
import os
import sys
import shutil
from argparse import ArgumentParser
import logging

from convlab.util.custom_util import init_logging

project_path = (os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)


# ================================================================================================
# SERVER BEHAVIOUR
# ================================================================================================


def make_request_handler_class(dialServer):
    """
    """

    class RequestHandler(http.server.BaseHTTPRequestHandler):
        '''
            Process HTTP Requests
            :return:
        '''

        def do_POST(self):
            '''
            Handle only POST requests. Please note that GET requests ARE NOT SUPPORTED! :)
            '''
            self.error_free = True  # boolean which can become False if we encounter a problem

            agent_id = None
            self.currentSession = None
            self.currentUser = None
            prompt_str = ''
            reply = {}

            # ------Get the "Request" link.
            logging.info('-' * 30)

            request = self.path[1:] if self.path.find(
                '?') < 0 else self.path[1:self.path.find('?')]

            logging.info(f'Request: ' + str(request))
            logging.info(f'POST full path: {self.path}')

            if not 'Content-Length' in self.headers:
                data_string = self.path[self.path.find('?') + 1:]
            else:
                data_string = self.rfile.read(
                    int(self.headers['Content-Length']))

            # contains e.g:  {"session": "voip-5595158237"}
            logging.info("Request Data:" + str(data_string))

            recognition_fail = True  # default until we confirm we have received data
            try:
                data = json.loads(data_string)  # ValueError
                self.currentSession = data["sessionID"]  # KeyError
                self.currentUser = data["userID"] # KeyError
            except Exception as e:
                logging.info(f"Not a valid JSON object (or object lacking info) received. {e}")
            else:
                recognition_fail = False

            if request == 'init':
                try:
                    user_id = data.get("userID", None)
                    task_id = data.get("taskID", None)
                    agent_id = dialServer.agent_factory.start_call(
                        session_id=self.currentSession, user_id=user_id, task_id=task_id)
                    reply = dialServer.prompt(
                        dialServer.agent_factory.willkommen_message, session_id=self.currentSession)
                except Exception as e:
                    self.error_free = False
                    logging.info(f"COULD NOT INIT SESSION, EXCEPTION: {e}")
                else:
                    logging.info(f"A new call has started. Session: {self.currentSession}")

            elif request == 'next':

                # Next step in the conversation flow
                # map session_id to agent_id

                try:
                    agent_id = dialServer.agent_factory.retrieve_agent(
                        session_id=self.currentSession)
                except Exception as e:  # Throws a ExceptionRaisedByLogger
                    self.error_free = False
                    logging.info(f"NEXT: tried to retrieve agent but: {e}")
                else:
                    logging.info(f"Continuing session: {self.currentSession} with agent_id {agent_id}")
                if self.error_free:
                    try:
                        userUtterance = data["text"]  # KeyError
                        user_id = data["userID"]
                        task_id = data.get("taskID", None)
                        logging.info(f"Received user utterance {userUtterance}")
                        prompt_str = dialServer.agent_factory.continue_call(
                            agent_id, user_id, userUtterance, task_id)

                        if(prompt_str == dial_server.agent_factory.ending_message):
                            reply = dialServer.prompt(
                                prompt_str, session_id=self.currentSession, isfinal=True)
                        else:
                            reply = dialServer.prompt(
                                prompt_str, session_id=self.currentSession, isfinal=False)
                    except Exception as e:
                        logging.info(f"NEXT: tried to continue call but {e}")
                else:
                    reply = None

            elif request == 'end':

                # Request to stop the session.

                logging.info("Received request to Clean Session ID from the VoiceBroker...:" + self.currentSession)

                self.error_free = False

                try:
                    agent_id = dialServer.agent_factory.end_call(
                        session_id=self.currentSession)
                except Exception as e:  # an ExceptionRaisedByLogger
                    logging.info(f"END: tried to end call but: {e}")

            # ------ Completed turn --------------

            # POST THE REPLY BACK TO THE SPEECH SYSTEM
            logging.info("Sending prompt: " + prompt_str + " to tts.")
            self.send_response(200)  # 200=OK W3C HTTP Standard codes
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            logging.info(reply)
            try:
                self.wfile.write(reply.encode('utf-8'))
            except Exception as e:
                logging.info(f"wanted wo wfile.write but {e}")
                reply = dialServer.prompt(f"Error: I am sorry, we are working on this.", session_id=self.currentSession, isfinal=False)
                self.wfile.write(reply.encode('utf-8'))
            logging.info(dialServer.agent_factory.session2agent)
    return RequestHandler


# ================================================================================================
# DIALOGUE SERVER
# ================================================================================================

class DialogueServer(object):

    '''
     This class implements an HTTP Server
    '''

    def __init__(self, configPath):
        """    HTTP Server
        """

        configparser = ConfigParser()
        configparser.read(configPath)
        host = (configparser.get("GENERAL", "host"))
        task_file = (configparser.get("GENERAL", "task_file"))
        port = int(configparser.get("GENERAL", "port"))
        agentPath = (configparser.get("AGENT", "agentPath"))
        agentClass = (configparser.get("AGENT", "agentClass"))
        dialogueSave = (configparser.get("AGENT", "dialogueSave"))
        saveFlag = True

        if configparser.get("AGENT", "saveFlag") == "True":
            saveFlag = True

        mod = __import__(agentPath, fromlist=[agentClass])
        klass = getattr(mod, agentClass)
        self.host = host
        self.port = port
        self.agent_factory = AgentFactory(configPath, dialogueSave, saveFlag, task_file)

        shutil.copy(configPath, self.agent_factory.filepath)
        shutil.copy(agentPath.replace(".", "/") + ".py", self.agent_factory.filepath)

        logging.info("Server init")

    def run(self):
        """Listen to request in host dialhost and port dialport"""

        RequestHandlerClass = make_request_handler_class(self)

        server = http.server.HTTPServer(
            (self.host, self.port), RequestHandlerClass)
        logging.info(f'Server starting {self.host}:{self.port} (level=info)')

        try:
            while 1:
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            logging.info(f'Server stopping {self.host}:{self.port}')
            server.server_close()

            self.agent_factory.power_down_factory()

    def prompt(self, prompt, session_id, isfinal=False):
        '''
        Create a prompt, for the moment the arguments are

        :param prompt: the text to be prompt
        :param isfinal:  if it is the final sentence before the end of dialogue
        :return: reply in json
        '''

        reply = {}
        reply["sessionID"] = session_id
        reply["version"] = "0.1"
        reply["terminal"] = isfinal
        reply['sys'] = self._clean_text(prompt)
        reply["timeStamp"] = datetime.now().isoformat()

        logging.info(reply)
        return json.dumps(reply, ensure_ascii=False)

    def _clean_text(self, RAW_TEXT):
        """
        """
        # The replace() is because of how words with ' come out of the Template SemO.
        JUNK_CHARS = ['(', ')', '{', '}', '<', '>', '"', "'"]
        return ''.join(c for c in RAW_TEXT.replace("' ", "") if c not in JUNK_CHARS)


def save_log_to_file():
    import time

    dir_name = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'dialogueServer_LOG')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_file = open(os.path.join(
        dir_name, f"stdout_{current_time}.txt"), 'w')
    sys.stdout = output_file
    sys.stderr = output_file


# ================================================================================================
# MAIN FUNCTION
# ================================================================================================
if __name__ == "__main__":

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), 'info')

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./convlab/dialcrowd_server/agents/ddpt_agent.cfg",
                        help="path of server config file to load")
    args = parser.parse_args()

    # save_log_to_file()
    logging.info(f"Config-file being used: {args.config}")

    dial_server = DialogueServer(args.config)
    dial_server.run()

# END OF FILE
