# -*- coding: utf-8 -*-
# Copyright 2024 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SetSUMBT logging utilities."""

import os
import logging
import logging.config
import json
from datetime import datetime, timezone
import smtplib

LOG_RECORD_BUILTIN_ATTRS = {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname",
                            "levelno", "lineno", "module", "msecs", "message", "msg", "name", "pathname", "process",
                            "processName", "relativeCreated", "stack_info", "thread", "threadName", "taskName",
                            "send_email"}


def get_logger(logging_path: str,
               config_name: str = "logger_config",
               send_emails: bool = False,
               email_config: str = "email_config",
               email_subject: str = "SetSUMBT Run Log") -> logging.Logger:
    """
    Get a logger object.

    Args:
        logging_path: The path to the log file.
        config_name: The name of the config file.
        send_emails: Whether to send emails.
        email_config: The name of the email config file.
        email_subject: The subject of the email.

    Returns:
        The logger object.
    """

    logger = logging.getLogger("__main__")

    # Load the configuration
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs', f"{config_name}.json")
    with open(path, 'r') as reader:
        config = json.load(reader)
        reader.close()

    # Set the logging path
    for name, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = logging_path + handler["filename"]
        config["handlers"][name] = handler

    # Set the email config
    if not send_emails:
        email_config = None
        config["handlers"].pop("email", None)
        if 'email' in config['loggers']['root']['handlers']:
            config['loggers']['root']['handlers'].remove('email')

    if email_config is not None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs',
                            f"{email_config}.json")
        with open(path, 'r') as reader:
            email_config = json.load(reader)
            reader.close()

        config['handlers']['email']['mailhost'] = (email_config['mailhost'], email_config['port'])
        address = f"{email_config['username']}@hhu.de"
        config['handlers']['email']['fromaddr'] = address
        config['handlers']['email']['toaddrs'] = [address]
        config['handlers']['email']['subject'] = email_subject
        config['handlers']['email']['credentials'] = (email_config['username'], email_config['password'])

    logging.config.dictConfig(config)

    return logger


class JSONLFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        message = {
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%H:%M %m-%d-%y"),
        }

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message

class LogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_message(record)
        return message

    def _prepare_log_message(self, record: logging.LogRecord):
        message_dict = {
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%H:%M %m-%d-%y"),
        }

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message_dict[key] = val

        message = f"{message_dict['timestamp']} - {message_dict['message']}"
        for key, item in message_dict.items():
            if key not in ["message", "timestamp"]:
                message += f"\n\t\t\t{key}: {item}"

        return message


class SSLSMTPHandler(logging.handlers.SMTPHandler):
    def emit(self, record):
        """
        Emit a record.
        """
        try:
            send_email = record.send_email
        except AttributeError:
            send_email = False

        if send_email:
            self._send_email(record)

    def _format_message(self, record):
        msg = self.format(record)
        msg = f"From: {self.fromaddr}\nTo: {self.toaddrs}\nSubject: {self.subject}\n\n{msg}"
        return msg

    def _send_email(self, record):
        try:
            message = self._format_message(record)

            port = self.mailport
            if not port:
                port = smtplib.SMTP_PORT
            smtp = smtplib.SMTP_SSL(self.mailhost, port)

            if self.username:
                smtp.login(self.username, self.password)
            smtp.sendmail(self.fromaddr, self.toaddrs, message)
            smtp.quit()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
