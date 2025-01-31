import os
from datetime import datetime
import torch

def make_saved_dir(saved_dir):
    """
    :param saved_dir:
    :return: {saved_dir}/{%m-%d-%H-%M-%S}
    """
    print("saved_dir:",saved_dir)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d-%H-%M-%S'))
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)
    return saved_dir


def log_to_file(file, **kwargs):
    with open(file, 'a') as f:
        f.write(','.join([f'{k}={v}' for k, v in kwargs.items()]))
        f.write('\n')