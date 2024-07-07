# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging

import numpy as np
import random
import torch
from datetime import datetime

import platform
import subprocess
import yaml
from pytz import timezone, utc

import psutil

def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass

def get_datetime_str(add_random_str=False):
    """Get string based on current datetime."""
    datetime_str = datetime.now(timezone('EST')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    if add_random_str:
        # Add a random integer after the datetime string
        # This would largely decrease the probability of having the same directory name
        # when running many experiments concurrently
        return '{}_{}'.format(datetime_str, np.random.randint(low=1, high=10000))
    else:
        return datetime_str

def get_system_info():
    # Code modified from https://stackoverflow.com/a/58420504
    try:
        info = {}
        info['git_revision_hash'] = get_git_revision_hash()
        info['platform'] = platform.system()
        info['platform-release'] = platform.release()
        info['platform-version'] = platform.version()
        info['architecture'] = platform.machine()
        info['processor'] = platform.processor()
        info['ram'] = '{} GB'.format(round(psutil.virtual_memory().total / (1024.0 **3)))
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_count'] = psutil.cpu_count()

        # Calculate percentage of available memory
        # Referred from https://stackoverflow.com/a/2468983
        info['percent_available_ram'] = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        return info
    except Exception as e:
        return None

def get_git_revision_hash():
    # Referred from https://stackoverflow.com/a/21901260
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())[2:-1]
    except:
        return ''

def load_yaml_config(path):
    """Load the config file in yaml format.
    Args:
        path (str): Path to load the config file.
    Returns:
        dict: config.
    """
    with open(path, 'r') as infile:
        # return yaml.safe_load(infile)
        return yaml.load(infile, Loader=yaml.Loader)


def save_yaml_config(config, path):
    """Load the config file in yaml format.
    Args:
        config (dict object): Config.
        path (str): Path to save the config.
    """
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def setup(args):
  
    # Set the working directory
    args.work_dir = os.path.join('experiments',
                                args.dataset_name,
                                args.exp_id,
                                get_datetime_str(add_random_str=True))

    # Set the random seed
    set_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    # Setup logging
    logfile = os.path.join(args.work_dir, 'logging.log')
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=logging.INFO, datefmt='%m-%d %H:%M')
    file_stream = open(logfile, 'w')
    handlers = [logging.StreamHandler(file_stream)]

    # Get and save system info
    system_info = get_system_info()
    if system_info is not None:
        save_yaml_config(system_info, path='{}/system_info.yaml'.format(args.work_dir))
        logging.info("The system information is well saved.")

    # Save the config file
    if args.model != 'all':
        args_add_path = os.path.join('utils',
                                    'args_add',
                                    args.model+'.yaml')

        args_add = load_yaml_config(args_add_path)
        args.lr = args_add[args.dataset_name]['lr']
        args.n_epochs = args_add['n_epochs']

    if args is not None:
        save_yaml_config(vars(args), path='{}/args_info.yaml'.format(args.work_dir))
        logging.info("The args information is well saved.")

    # define a Handler which writes INFO messages or higher to the sys.stderr
    # handlers.append(logging.StreamHandler())

    for handler in handlers:
        handler.setLevel(logging.INFO)
        # tell the handler to use this format
        handler.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(handler)

        logging.info('Starting the expriment......')
        logging.info(':::::: Commandline args:\n%s', args)
