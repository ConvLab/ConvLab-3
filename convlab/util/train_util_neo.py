import os
import sys
import time
import logging
import logging.handlers
import torch
# from google.cloud import storage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LevelFilter(object):
    #This class is used for multi-level logging separation
    def __init__(self, level):
        self.level = logging._checkLevel(level)

    def filter(self, record):
        return record.levelno == self.level

def init_logging_handler(log_dir, extra='', logging_mode=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        '{}/log_{}.txt'.format(log_dir, current_time + extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging_mode)
    return current_time


def init_logging_nunu(save_dir, mode):

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join(save_dir, 'logs')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    if mode.lower() == 'debug':
        mode = logging.DEBUG
    elif mode.lower() == 'info':
        mode = logging.INFO
    elif mode.lower() == 'warning':
        mode = logging.WARNING
    elif mode.tolower() == 'error':
        mode = logging.ERROR

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(mode)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    levels = ['INFO', 'ERROR', 'WARNING', 'DEBUG']

    for lv in levels:
        lv_handler = logging.handlers.RotatingFileHandler(os.path.join(save_dir, 'train_' + lv + '.log'))
        lv_filter = LevelFilter(lv)
        lv_handler.addFilter(lv_filter)
        lv_handler.setFormatter(logFormatter)
        rootLogger.addHandler(lv_handler)
    #Create different file for each levels

    return current_time, save_dir

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data

# def save_to_bucket(bucket_name, bucket_file_path, local_file_path):

#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)
#     blob = bucket.blob(bucket_file_path)
#     blob.upload_from_filename(local_file_path)
