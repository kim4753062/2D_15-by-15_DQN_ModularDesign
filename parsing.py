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
2023-10-16
Problem definition:
Placing 3 production wells sequentially in 2D 15-by-15 heterogeneous reservoir
Period of well placement is 90 days, and total production period is 270 days (Well placement time from simulation starts: 0-90-180)
'''

'''
Deep Q Network (DQN) State, Action, Environment, Reward definition:

State: Pressure distribution, Oil saturation, Well placement map
Action: Well placement (Coordinate of well location)
Environment: Reservoir simulator
Reward: NPV at each action time segment
'''

'''
Directory setting: ([]: Folder)

- [Master directory] (Prerequisite directory)
--- Algorithm launcher code (.py) (Prerequisite file)
--- [Basic simulation data directory] ("data") (Prerequisite directory)
----- Simulation data template (Current simulator type: Eclipse, .DATA) (Prerequisite file)
----- Simulation permeability set file (.mat, .DATA, ...) (Prerequisite file)
--- [Figure directory] ("figure")
----- All well placement figure (.png): f"All Well placement-Step{num. of algorithm iteration}.png"
----- All NPV figure (.png): f"NPV-Step{num. of algorithm iteration}.png"
----- [Figure at each algorithm step dicrectory]: f"Step{num. of algorithm iteration}-WP"
------- [Well placement at each algorithm step]: "WellPlacement"
--------- Well placement figure for each sample (.png): f"Well placement-Step{num. of algorithm iteration}-Sample{sample number}.png"
------- [NPV at each algorithm step]: "NPV"
--------- NPV figure for each sample (.png): f"NPV-Step{num. of algorithm iteration}-Sample{sample number}.png"
--- [Training log directory] ("log")
----- Deep learning model training log file (.log): f"DQN_Training_Step{num. of algorithm iteration}.log"
--- [Deep learning model and Optimizer storage directory] ("model")
----- Deep learning model and Optimizer state file (.pkl): f"DQN_Step_{num. of algorithm iteration}.model"
--- [Simulation directory] ("simulation")
----- [Simulation sample directory #f"Step{num. of algorithm iteration}_Sample{sample number}"]
------- Simulation data file (.DATA): for each Well placement (== Action) timestep
------- Simulation include file (PERMX.DATA, WELL.DATA)
------- # File naming convention: f"{file type}_Sam{sample number}_Seq{timestep index}.DATA", Sam: Sample number (starts from 1), Seq: Time sequence (starts from 1)
--- [Variable storage directory] ("variables")
----- [Experiece sample storage directory] ("Experience_sample")
------- Experience sample file (.pkl): f"Experience_sample_{num. of algorithm iteration}.pkl"
----- [Simulation sample storage directory] ("Simulation_sample")
------- Simulation sample file (.pkl): f"Simulation_sample_{num. of algorithm iteration}.pkl"
'''

# Modified from J.Y. Kim. (2020)
# Arguments: Directory and File name
args.master_directory = os.getcwd()
args.basicfilepath = 'data'
args.simulation_directory = 'simulation'
args.variable_save_directory = 'variables'
args.deeplearningmodel_save_directory = 'model'
args.figure_directory = 'figure'
args.log_directory = 'log'
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
# args.random_seed = 202022673
args.random_seed = 123456789

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
args.policy_param_tracker = []
args.reward_unit = 10 ** 6 # For scaling

args.max_iteration = 20  # Maximum iteration num. of algorithm, MAX_STEPS
# args.max_iteration = 5  # Maximum iteration num. of algorithm, MAX_STEPS # 2023-10-10: PER TEST

args.sample_num_per_iter = 100  # Simulation sample num. of each iteration of algorithm
# args.sample_num_per_iter = 5  # Simulation sample num. of each iteration of algorithm # 2023-10-10: PER TEST
args.experience_num_per_iter = args.total_well_num_max * args.sample_num_per_iter  # Experience sample num. of each iteration of algorithm, h

args.replay_batch_num = 32  # Replay batch num., B

args.nn_update_num = 50  # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value
# args.nn_update_num = 5  # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value # 2023-10-10: PER TEST

args.batch_size = 64  # Batch size, N
# args.batch_size = 4  # Batch size, N # 2023-10-10: PER TEST

# 2023-10-04: Dynamic replay memory size (== round(70% of [max_iteration*sample_num_per_iter*action_num]))
args.replay_memory_size = round(args.max_iteration * args.sample_num_per_iter * args.total_well_num_max * 0.7)  # Replay memory size, K

args.discount_rate = 0.1  # Used for calculation of NPV
args.discount_factor = 0.5  # Used for Q-value update

args.input_flag = ('PRESSURE', 'SOIL', 'Well_placement')  # Data for State

# 2023-10-10: Prioritized Experience Replay (PER) option added
args.activate_PER = True
# args.activate_PER = False
if args.activate_PER == True:
    # "Foundations of Deep Reinforcement Learning" in Korean translation, pp. 124
    args.td_err_init = 1000 # Initial TD-error value for PER
    args.prob_compensation = 0.01 # Probability compensation factor for each experience when TD-error == 0 (epsilon)
    args.prob_exponent = 1 # Constant value for Priority exponent, eta
    args.prob_exponent_start = 0.1 # Start value for Priority exponent decaying with DQN step, eta
    args.prob_exponent_end = 1 # End value for Priority exponent decaying with DQN step, eta
