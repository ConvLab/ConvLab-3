import os
import sys
import time
import logging
import torch
# from google.cloud import storage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def init_logging_nunu(save_dir):

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join(save_dir, 'logs')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(save_dir, f'train.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    return current_time


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
