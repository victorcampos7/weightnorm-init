import os
import sys
import csv
import json
import numpy
import torch
import random
import logging
import datetime


def get_storage_dir():
    return "../logs"


def get_model_dir(model_name, storage_dir=None):
    storage_dir = storage_dir or get_storage_dir()
    return os.path.join(storage_dir, model_name)


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    numpy.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def get_log_path(model_dir):
    return os.path.join(model_dir, "log.txt")


def get_config_path(model_dir):
    return os.path.join(model_dir, "config.json")


def get_loggers(model_dir):
    formatter = logging.Formatter("%(message)s")

    def setup_logger(name, log_file, level=logging.INFO):
        """Function setup as many loggers as you want"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    path_log = get_log_path(model_dir)

    create_folders_if_necessary(path_log)

    return setup_logger('logs', path_log)


def save_config(args, model_dir):
    path = get_config_path(model_dir)
    create_folders_if_necessary(path)
    with open(path, 'wt') as fh:
        exp_json = json.dumps(vars(args), indent=2)
        fh.write(exp_json)
        print(exp_json)


def get_model_name(args):
    if args.model is not None:
        model_name = args.model
    else:
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        model_name = '%s-%s' % (args.dataset, suffix)
    if args.model_prefix:
        model_name = args.model_prefix + '/' + model_name
    return model_name


def get_csv_path(model_dir):
    return os.path.join(model_dir, "log.csv")


def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def get_model_path(model_dir, module):
    return os.path.join(model_dir, "%s.pt" % module)


def save_model(model, model_dir, module):
    path = get_model_path(model_dir, module)
    create_folders_if_necessary(path)
    torch.save(model.state_dict(), path)


def model_exists(model_dir, module):
    return os.path.exists(get_model_path(model_dir, module))


def load_model(model, model_dir, module):
    path = get_model_path(model_dir, module)
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    if hasattr(model, 'eval'):
        model.eval()


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.json")


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)


def load_status(model_dir):
    path = get_status_path(model_dir)
    with open(path) as file:
        return json.load(file)


def pretty_number(x):
    if x // 1e6 > 0:
        return "%.2fM" % float(x / 1e6)
    elif x // 1e3 > 0:
        return "%.2fk" % float(x / 1e3)
    else:
        return str(x)
