######################################## 1. Import required modules #############################################
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

import random
import numpy.random
import torch.random

from collections import deque
from copy import deepcopy

import os.path
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

# 2023-09-26: Legacy (Not be used)
# from datetime import datetime
# import copy
# import dill
# import shutil

######################################## 2. Define arguments #############################################
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

# 2023-05-02
# For Using GPU (CUDA)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2023-05-02
# For using tensorboard
args.istensorboard = False

'''
Problem definition:
Placing 5 production well sequentially in 2D 15-by-15 heterogeneous reservoir
Period of well placement is 120 days, and total production period is 600 days (Well placement time from simulation starts: 0-120-240-360-480)
'''

'''
Deep Q Network (DQN) State, Action, Environment, Reward definition:

State: Pressure distribution, Oil saturation, Well placement map
Action: Well placement (Coordinate of well location)
Environment: Reservoir simulator
Reward: NPV at each time segment
'''

'''
Directory setting: ([]: Folder)

- [Master directory] (Prerequisite directory)
--- Algorithm launcher code (.py, .ipynb, ...) (Prerequisite file)

--- [Basic simulation data directory] (data) (Prerequisite directory)
----- Simulation data template (Current simulator type: Eclipse, .DATA) (Prerequisite file)
----- Simulation permeability set file (.mat, .DATA, ...) (Prerequisite file)

--- [Simulation directory] (simulation)
----- [Simulation sample directory #f"Step{num. of algorithm iteration}_Sample{sample number}"]
------- Simulation data file (.DATA): for each Well placement timestep
------- Simulation include file (PERMX.DATA, WELL.DATA)
------- # File naming convention: f"{file type}_Sam{sample number}_Seq{timestep index}.DATA", Sam: Sample number (starts from 1), Seq: Time sequence (starts from 1)

--- [Variable storage directory] (Variables)
----- Variable storage.pkl
----- Global variable storage.dill

--- [Deep learning model storage directory] (DL Model)
----- Deep learning model.pkl
'''

# Modified from J.Y. Kim. (2020)
# Arguments: Directory and File name
args.master_directory = os.getcwd()
args.basicfilepath = 'data'
args.simulation_directory = 'simulation'
args.variable_save_directory = 'variables'
args.deeplearningmodel_save_directory = 'model'
args.figure_directory = 'figure'
args.ecl_filename = '2D_ECL'
args.perm_filename = 'PERMX'
args.well_filename = 'WELL'

# Arguments: Reservoir simulation
args.gridnum_x = 15
args.gridnum_y = 15
args.gridsize_x = 180  # ft
args.gridsize_y = 180  # ft

args.time_step = 90  # days
args.total_production_time = 270  # days

args.prod_well_num_max = 3
args.inj_well_num_max = 0
args.total_well_num_max = args.prod_well_num_max + args.inj_well_num_max

args.initial_PRESSURE = 3500  # psi
args.initial_SOIL = 0.75

# Arguments: Random seed number
args.random_seed = 202022673

# Arguments: Price and Cost of oil and water
args.oil_price = 60  # $/bbl
args.water_treatment = 3  # $/bbl
args.water_injection = 5  # $/bbl

# Arguments: Hyperparameters for Deep Q Network (DQN)
args.learning_rate = 0.001  # Learning rate Alpha
# 2023-10-04: Policy parameter (Tau for Boltzmann policy, Epsilon for Epsilon-greedy policy) were unified
args.policy_type = "e-Greedy"
args.policy_param_start = 1.0
args.policy_param_end = 0.1
# args.policy_param_tracker = [args.policy_param_start]
args.policy_param_tracker = []
args.reward_unit = 10 ** 6 # For scaling

# args.max_iteration = 20  # Maximum iteration num. of algorithm, MAX_STEPS
args.max_iteration = 50  # 2023-10-04: Just for test

args.sample_num_per_iter = 100  # Simulation sample num. of each iteration of algorithm
args.experience_num_per_iter = args.total_well_num_max * args.sample_num_per_iter  # Experience sample num. of each iteration of algorithm, h

args.replay_batch_num = 32  # Replay batch num., B
# args.replay_batch_num = 4  # For debugging

args.nn_update_num = 50  # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value
# args.nn_update_num = 10  # For debugging

args.batch_size = 64  # Batch size, N

# 2023-10-04: Dynamic replay memory size (== round(70% of [max_iteration*sample_num_per_iter*action_num]))
# args.replay_memory_size = 5000  # Replay memory size, K
args.replay_memory_size = round(args.max_iteration * args.sample_num_per_iter * args.total_well_num_max * 0.7)  # Replay memory size, K

args.discount_rate = 0.1  # Used for calculation of NPV
# args.discount_factor = 1  # Used for Q-value update
args.discount_factor = 0.5  # Used for Q-value update

args.input_flag = ('PRESSURE', 'SOIL', 'Well_placement')  # Data for State