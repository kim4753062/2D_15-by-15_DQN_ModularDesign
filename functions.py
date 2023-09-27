from objects import *

import random
import numpy as np

import os.path
from copy import deepcopy

import matplotlib.pyplot as plt

################################################################################################
#################################### Definition: Functions #####################################
################################################################################################

############################## Reading Eclipse Dynamic Data (.PRT) ############################# Certified
# algorithm_iter_count: algorithm iteration num. (m)
# sample_num: sample num. (1 ~ args.sample_num_per_iter)
# tstep_idx: sample time step index
# filename: simulation file name
# dynamic_type: dynamic data type to collect ('PRESSURE' or 'SOIL')
def _read_ecl_prt_2d(args, algorithm_iter_count: int, sample_num: int, tstep_idx: int, dynamic_type: str) -> list:
    # Check if dynamic type input is (1) 'PRESSURE', (2) 'SOIL'
    if not dynamic_type in ['PRESSURE', 'SOIL']:
        print("Assign correct dynamic data output type!: 'PRESSURE', 'SOIL'")
        return -1

    # File IO
    # 1. Open .PRT file
    with open(f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx}.PRT") as file_read:
        line = file_read.readline()
        if dynamic_type == 'PRESSURE':
            # 2. Find the location of dynamic data (PRESSURE case)
            # 2023-08-19: consideration for 2-digit timestep
            if args.time_step * tstep_idx < 100:
                while not line.startswith(f"  {dynamic_type} AT    {args.time_step * tstep_idx}"):
                    line = file_read.readline()
            elif args.time_step * tstep_idx >= 100:
                while not line.startswith(f"  {dynamic_type} AT   {args.time_step * tstep_idx}"):
                    line = file_read.readline()
            # 3. Dynamic data starts from 10th line below the line ["  {dynamic_type} AT   {args.time_step * tstep_idx}"]
            for i in range(1,10+1):
                line = file_read.readline()
            # 4. Collect dynamic data
            lines_converted = []
            for i in range(1, args.gridnum_y+1):
                lines_converted.append([element.strip() for element in line.split()][3::])
                line = file_read.readline()
        elif dynamic_type == 'SOIL':
            # 2. Find the location of dynamic data (SOIL case)
            # 2023-08-19: consideration for 2-digit timestep
            if args.time_step * tstep_idx < 100:
                while not line.startswith(f"  {dynamic_type}     AT    {args.time_step * tstep_idx}"):
                    line = file_read.readline()
            elif args.time_step * tstep_idx >= 100:
                while not line.startswith(f"  {dynamic_type}     AT   {args.time_step * tstep_idx}"):
                    line = file_read.readline()
            # 3. Dynamic data starts from 10th line below the line ["  {dynamic_type}     AT   {args.time_step * tstep_idx}"]
            for i in range(1,10+1):
                line = file_read.readline()
            # 4. Collect dynamic data
            lines_converted = []
            for i in range(1, args.gridnum_y+1):
                lines_converted.append([element.strip() for element in line.split()][3::])
                line = file_read.readline()

    # 5. Post-processing (String replacement from (1) '*' to '.', (2) String to Float (Only for 2D)
    for i in range(len(lines_converted)):
        for j in range(len(lines_converted[i])):
            lines_converted[i][j] = float(lines_converted[i][j].replace('*', '.'))

    return lines_converted

################## Reading Eclipse Production or Injection Data (.RSM) ################### Certified
# algorithm_iter_count: algorithm iteration num. (m)
# sample_num: sample num. (1 ~ args.sample_num_per_iter)
# tstep_idx: sample time step index
# filename: simulation file name
# data_type: Production or Injection result data type ('FOPT', 'FWPT', 'FWIT')
def _read_ecl_rsm(args, algorithm_iter_count: int, sample_num: int, tstep_idx: int, dynamic_type: str) -> list:
    # Check if data type input is (1) 'FOPT', (2) 'FWPT', (3) 'FWIT'
    if not dynamic_type in ['FOPT', 'FWPT', 'FWIT']:
        print("Assign correct output data type!: 'FOPT', 'FWPT', 'FWIT'")
        return -1

    # File IO
    # 1. Open .RSM file
    # with open(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}", f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx}.RSM")) as file_read:
    with open(f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx+1}.RSM") as file_read:
        line = file_read.readline()
        # 2. Find the location of simulation result data
        while not line.startswith(f" TIME"):
            line = file_read.readline()
        # 3. 1st time of simulation result data starts from 6th line below the line [" TIME"]
        for i in range(1,5+1):
            line = file_read.readline()
        # 4. Collect production or injection data
        lines_converted = []
        if dynamic_type == 'FOPT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][2])
                line = file_read.readline()
        elif dynamic_type == 'FWPT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][3])
                line = file_read.readline()
        elif dynamic_type == 'FWIT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][4])
                line = file_read.readline()

    # 5. Post-processing (String replacement from (1) '*' to '.', (2) String to Float)
    for i in range(len(lines_converted)):
        lines_converted[i] = float(lines_converted[i].replace('*', '.'))

    return lines_converted

################################### Run ECL Simulator #################################### Certified
# program: 'eclipse' or 'frontsim'
# filename: simulation file name (ex. 2D_ECL_Sam1_Seq2)
def _run_program(args, program: str, filename: str):
    # Check if dynamic type input is (1) 'eclipse', (2) 'frontsim'
    if not program in ['eclipse', 'frontsim']:
        print("Use correct simulator exe file name!: 'eclipse', 'frontsim'")
        return -1
    command = fr"C:\\ecl\\2009.1\\bin\\pc\\{program}.exe {filename} > NUL"
    os.system(command)

#################################### Boltzmann policy #################################### Certified
# Transform Q-value map (2D array) to Well placement probability map (2D array)
def _Boltzmann_policy(args, Q_value: list, well_placement: list) -> list:
    Q_value_list = np.squeeze(Q_value)
    exp_tau = deepcopy(Q_value_list)
    probability = deepcopy(Q_value_list)

    # Prevent overflow error by exponential operation
    max_Q_value = np.array(Q_value_list).flatten().max()

    # Get exponential of all elements in Q_value
    for i in range(0, args.gridnum_y):
        for j in range(0, args.gridnum_x):
            exp_tau[i][j] = np.exp((exp_tau[i][j]-max_Q_value)/args.tau)

    # Calculate probability map
    for i in range(0, args.gridnum_y):
        for j in range(0, args.gridnum_x):
            probability[i][j] = exp_tau[i][j] / np.concatenate(np.array(exp_tau)).sum()

    # Mask probability map: Setting probability = 0 where wells were already exists,
    # and Scale the rest of probability map
    probability = [[0 if well_placement[i][j] != 0 else probability[i][j] for j in range(args.gridnum_x)] for i in range(args.gridnum_y)]
    probability = [[(probability[i][j]/np.concatenate(np.array(probability)).sum()) for j in range(args.gridnum_x)] for i in range(args.gridnum_y)]

    return probability

#################################### Greedy policy ####################################
# Return well location from Q-value map with greedy policy
def _Greedy_policy(args, Q_value: list, well_placement: list) -> tuple:
    Q_value_list = np.squeeze(Q_value)
    Q_value_list_mask = deepcopy(Q_value_list)

    # Masking well places that they already exist
    for row in range(args.gridnum_y):
       for col in range(args.gridnum_x):
           if well_placement[row][col] == 1:
               Q_value_list_mask[row][col] = np.NINF  # (x, y) for ECL, (Row(y), Col(x)) for Python / 2D-map array

    max_row, max_col = np.where(np.array(Q_value_list_mask) == max(map(max, np.array(Q_value_list_mask))))
    well_loc = (max_col[0]+1, max_row[0]+1)

    return well_loc

# #################################### e-Greedy policy ####################################
# Return well location from Q-value map with e-Greedy policy
def _epsilon_Greedy_policy(args, Q_value: list, well_placement: list) -> tuple:
    Q_value_list = np.squeeze(Q_value)
    Q_value_list_mask = deepcopy(Q_value_list)

    epsilon = args.epsilon
    exploration_or_exploitation = random.random()

    # Masking well places that they already exist
    for row in range(args.gridnum_y):
       for col in range(args.gridnum_x):
           if well_placement[row][col] == 1:
               Q_value_list_mask[row][col] = np.NINF  # (x, y) for ECL, (Row(y), Col(x)) for Python / 2D-map array

    if epsilon > exploration_or_exploitation: # Exploration, random.randint(a,b): Return random integer in [a, b] (a<=X<=b)
        max_row, max_col = random.randint(1, args.gridnum_y), random.randint(1, args.gridnum_x)
        well_loc = (max_col, max_row)
    elif epsilon <= exploration_or_exploitation: # Exploitation
        max_row, max_col = np.where(np.array(Q_value_list_mask) == max(map(max, np.array(Q_value_list_mask))))
        well_loc = (max_col[0]+1, max_row[0]+1)
    else:
        print("Well location selection was not appropriately done for epsilon-Greedy policy!")

    return well_loc

#################################### Action Selection #################################### Certified
# Select well location from well placement probability map (Generated by Boltzmann policy)
def _select_well_loc(args, probability: list) -> tuple:
    # Create cumulative probability function with given probability map by policy
    cumsum_prob = np.cumsum(probability)
    CDF = np.append([0], cumsum_prob)

    # Generate random number (0~1)
    CDF_prob = random.random()

    # Find corresponding well location
    for i in range(0, len(CDF)-1):
        if (CDF_prob >= CDF[i]) and (CDF_prob < CDF[i+1]):
            well_loc = ((i%args.gridnum_x)+1, (i//args.gridnum_x)+1) # (x, y) for ECL, (Row, Col) for Python.
            return well_loc

    # If well location selection failed.
    print("Well location selection was not appropriately done for Boltzmann policy!")

#################################### NPV Calculation #####################################
def _calculate_income(args, tstep_idx: int, FOPT: list, FWPT: list, FWIT: list) -> float:
    # Calculate income from [tstep_idx] to [tstep_idx+1]
    # e.g. tstep_idx == 0 >> income of 0 ~ 120 day, tstep_idx == 1 >> income of 120 ~ 240 day
    oil_income = (FOPT[tstep_idx+1] - FOPT[tstep_idx]) * args.oil_price / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))
    water_treat = (FWPT[tstep_idx+1] - FWPT[tstep_idx]) * args.water_treatment / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))
    water_inj = (FWIT[tstep_idx+1] - FWIT[tstep_idx]) * args.water_injection / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))

    income = oil_income - water_treat - water_inj
    income = income / (args.reward_unit) # 2023-07-17: for test of scale of reward.

    return income

############################ Generating Simulation Data File #############################
# Need to utilize RESTART Option
def _ecl_data_generate(args, algorithm_iter_count: int, sample_num: int, timestep: int, well_loc_list: list[tuple]) -> None:
    output_data_file = []
    output_perm_file = []
    output_well_file = []

    # Read and modify simulation data template on Python
    with open(f"{os.path.join(args.basicfilepath, args.ecl_filename)}.template", 'r') as file_read_data:
        line = file_read_data.readline()
        output_data_file.append(line)
        while not line.startswith("[#PERMX]"):
            line = file_read_data.readline()
            output_data_file.append(line)
        line = line.replace("[#PERMX]", f"\'{args.perm_filename}_Sam{sample_num}_Seq{timestep}.DATA\'")
        output_data_file[-1] = line
        while not line.startswith("[#WELL]"):
            line = file_read_data.readline()
            output_data_file.append(line)
        line = line.replace("[#WELL]", f"\'{args.well_filename}_Sam{sample_num}_Seq{timestep}.DATA\'")
        output_data_file[-1] = line
        while line:
            line = file_read_data.readline()
            output_data_file.append(line)

    # Read Permeability distribution file
    with open(f"{os.path.join(args.basicfilepath, args.perm_filename)}.DATA", 'r') as file_read_perm:
        line = file_read_perm.readline()
        output_perm_file.append(line)
        while line:
            line = file_read_perm.readline()
            output_perm_file.append(line)

    # Write simulation main data and include files
    sample_simulation_directory = f"Step{algorithm_iter_count}_Sample{sample_num}"
    sample_data_name = f"{args.ecl_filename}_Sam{sample_num}_Seq{timestep}.DATA"
    sample_perm_name = f"{args.perm_filename}_Sam{sample_num}_Seq{timestep}.DATA"
    sample_well_name = f"{args.well_filename}_Sam{sample_num}_Seq{timestep}.DATA"

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_data_name)}", 'w') as file_write_data:
        for i in range(len(output_data_file)):
            file_write_data.write(output_data_file[i])

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_perm_name)}", 'w') as file_write_perm:
        for i in range(len(output_perm_file)):
            file_write_perm.write(output_perm_file[i])

    for i in range(len(well_loc_list)):
        output_well_file.append(f"--WELL #{i+1}\n"
                                f"WELSPECS\n P{i+1} ALL {well_loc_list[i][0]} {well_loc_list[i][1]} 1* LIQ 3* NO /\n/\n \n"
                                f"COMPDAT\n P{i+1} {well_loc_list[i][0]} {well_loc_list[i][1]} 1 1 1* 1* 1* 1 1* 1* 1* Z /\n/\n \n"
                                f"WCONPROD\n P{i+1} 1* BHP 5000 4* 1500.0 /\n/\n \n"
                                f"TSTEP\n 1*{args.time_step} /\n \n \n")

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_well_name)}", 'w') as file_write_well:
        for i in range(len(output_well_file)):
            file_write_well.write(output_well_file[i])

################################### Simulation Sampler ###################################
# Make One full Well placement sample
def _simulation_sampler(args, algorithm_iter_count: int, sample_num: int, network, policy: str) -> WellPlacementSample:
    well_placement_sample = WellPlacementSample(args=args)

    Q_network = network

    # Read simulation samples if they already exist
    if os.path.exists(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}", f"{args.ecl_filename}_SAM{sample_num}_SEQ{args.total_well_num_max}.PRT")):
        # Read well location from simulation file
        well_loc_list_from_simulation = []

        with open(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}", f"{args.well_filename}_Sam{sample_num}_Seq{args.total_well_num_max}.DATA")) as file_read:
            line = file_read.readline()
            for time_step in range(0, args.total_well_num_max):
                line = file_read.readline()
                line = file_read.readline()
                line_list = [element.strip() for element in line.split()]
                well_loc = (int(line_list[2]), int(line_list[3]))
                well_loc_list_from_simulation.append(well_loc)
                if time_step == args.total_well_num_max - 1:
                    break
                while not line.startswith("--WELL"):
                    line = file_read.readline()

        for time_step in range(0, args.total_well_num_max):
            os.chdir(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}"))

            well_loc = well_loc_list_from_simulation[time_step]
            well_placement_sample.well_loc_list.append(well_loc)

            # Read PRESSURE, SOIL map from .PRT file and calculate income with .RSM file
            pressure_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='PRESSURE')
            soil_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='SOIL')

            fopt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FOPT')
            fwpt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWPT')
            fwit = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWIT')

            income = _calculate_income(args=args, tstep_idx=time_step, FOPT=fopt, FWPT=fwpt, FWIT=fwit)

            well_placement_map = deepcopy(well_placement_sample.well_loc_map[time_step])
            for i in range(0, args.gridnum_x):
                for j in range(0, args.gridnum_y):
                    if (i == well_loc[0]-1) and (j == well_loc[1]-1):
                        well_placement_map[j][i] = 1

            # Append PRESSURE map, SOIL map, Well placement map, Income
            well_placement_sample.PRESSURE_map.append(pressure_map)
            well_placement_sample.SOIL_map.append(soil_map)
            well_placement_sample.well_loc_map.append(well_placement_map)
            well_placement_sample.income.append(income)

            os.chdir('../../')

        _visualization_sample(args=args, well_placement_sample=well_placement_sample, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num)

        return well_placement_sample

    # If simulation samples don't exist, do Well placemnet sampling
    if not os.path.exists(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}")):
        os.mkdir(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}"))

    for time_step in range(0, args.total_well_num_max):
        # Inference of Q-value from PRESSURE, SOIL, and Well placement
        Q_map = Q_network.forward(torch.tensor(data = [[well_placement_sample.PRESSURE_map[time_step], well_placement_sample.SOIL_map[time_step], well_placement_sample.well_loc_map[time_step]]], dtype=torch.float, device='cuda', requires_grad=True))

        # Calculate well placement probability and Specify well location
        if policy == "Boltzmann":
            prob = _Boltzmann_policy(args=args, Q_value=Q_map.tolist(), well_placement=well_placement_sample.well_loc_map[time_step])
            well_loc = _select_well_loc(args=args, probability=prob)
            well_placement_sample.well_loc_list.append(well_loc)
        elif policy == "Greedy":
            well_loc = _Greedy_policy(args=args, Q_value=Q_map.tolist(), well_placement=well_placement_sample.well_loc_map[time_step])
            well_placement_sample.well_loc_list.append(well_loc)
        elif policy == "e-Greedy":
            well_loc = _epsilon_Greedy_policy(args=args, Q_value=Q_map.tolist(), well_placement=well_placement_sample.well_loc_map[time_step])
            well_placement_sample.well_loc_list.append(well_loc)
        else:
            print("Use proper policy!: Boltzmann, Greedy, or e-Greedy")

        # Generate and run simulation file
        _ecl_data_generate(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, timestep=time_step+1, well_loc_list=well_placement_sample.well_loc_list)

        os.chdir(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}"))
        _run_program(args=args, program='eclipse', filename=f"{args.ecl_filename}_Sam{sample_num}_Seq{time_step+1}")

        # Read PRESSURE, SOIL map from .PRT file and calculate income with .RSM file
        pressure_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='PRESSURE')
        soil_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='SOIL')

        fopt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FOPT')
        fwpt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWPT')
        fwit = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWIT')

        income = _calculate_income(args=args, tstep_idx=time_step, FOPT=fopt, FWPT=fwpt, FWIT=fwit)

        well_placement_map = deepcopy(well_placement_sample.well_loc_map[time_step])
        for i in range(0, args.gridnum_x):
            for j in range(0, args.gridnum_y):
                if (i == well_loc[0]-1) and (j == well_loc[1]-1):
                    well_placement_map[j][i] = 1

        # Append PRESSURE map, SOIL map, Well placement map, Income
        well_placement_sample.PRESSURE_map.append(pressure_map)
        well_placement_sample.SOIL_map.append(soil_map)
        well_placement_sample.well_loc_map.append(well_placement_map)
        well_placement_sample.income.append(income)

        os.chdir('../../')

    _visualization_sample(args=args, well_placement_sample=well_placement_sample, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num)

    return well_placement_sample

################################### Experience Sampler ###################################
# Collect and save Experience instances from simulation samples
def _experience_sampler(args, simulation_sample_list: list[WellPlacementSample])-> list[Experience]:
    # 1. Save all Experience instances from simulation samples (experience_list)
    # 2. Pick random experience from experience_list
    experience_list = []

    for i in range(0, args.sample_num_per_iter):
        for j in range(0, args.total_well_num_max):
            exp = Experience(args=args)
            # 2023-07-09: Min-Max scaling for Pressure and SOIL
            exp.current_state = [np.array(simulation_sample_list[i].PRESSURE_map[j])/args.initial_PRESSURE, np.array(simulation_sample_list[i].SOIL_map[j])/args.initial_SOIL, simulation_sample_list[i].well_loc_map[j]]
            exp.current_action = simulation_sample_list[i].well_loc_list[j]
            exp.reward = simulation_sample_list[i].income[j]
            exp.next_state = [np.array(simulation_sample_list[i].PRESSURE_map[j+1])/args.initial_PRESSURE, np.array(simulation_sample_list[i].SOIL_map[j+1])/args.initial_SOIL, simulation_sample_list[i].well_loc_map[j+1]]
            exp.transform()
            experience_list.append(exp)

    experience_sample = random.sample(experience_list, args.experience_num_per_iter)

    return experience_sample

################ Visualization of NPV and Well Placement for each simulation samples ################
def _visualization_sample(args, well_placement_sample: WellPlacementSample, algorithm_iter_count: int, sample_num: int) -> None:
    # # 1. Well placement visualization
    coord_x = []
    coord_y = []

    for i in range(len(well_placement_sample.well_loc_list)):
        coord_x.append(well_placement_sample.well_loc_list[i][0]-1) # 0 ~ args.gridnum_x
        coord_y.append(well_placement_sample.well_loc_list[i][1]-1) # 0 ~ args.gridnum_y

    if not os.path.exists(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP')):
        os.mkdir(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP'))

    if not os.path.exists(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'WellPlacement')):
        os.mkdir(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'WellPlacement'))
    well_num = [i for i in range(1, args.total_well_num_max+1)]
    plt.figure(figsize=(8, 6))  # Unit: inch
    plt.tight_layout()
    plt.scatter(coord_x, coord_y, c='k')
    for i, txt in enumerate(well_num):
        plt.gca().text(coord_x[i] + 0.3, coord_y[i] + 0.3, txt, fontsize=10)
    plt.imshow(np.log(np.array(args.perm_field)).reshape(15, 15), cmap='jet')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("ln(Perm)")
    plt.gca().xaxis.tick_top()
    plt.gca().set_xticks(range(0, 15))
    plt.gca().set_xlabel("Grid X", loc='center')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_yticks(range(0, 15))
    plt.gca().set_ylabel("Grid Y")
    plt.gca().set_title(f"ln(Perm) map with well location (Step #{algorithm_iter_count}, Sample #{sample_num})", font="Arial", fontsize=16)
    figname = f'Well placement-Step{algorithm_iter_count}-Sample{sample_num}' + '.png'
    plt.savefig(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'WellPlacement', figname))
    plt.close()

    # # 2. NPV visualization
    if not os.path.exists(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'NPV')):
        os.mkdir(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'NPV'))
    npv_list = list(np.cumsum(np.array(well_placement_sample.income)) * args.reward_unit)

    time_step_list = [args.time_step * i for i in range(args.total_well_num_max + 1)]
    plt.figure(figsize=(8, 6))  # Unit: inch
    plt.plot(time_step_list, np.array([0] + npv_list) / (10 ** 6)) # Unit: MM$
    plt.gca().set_xlim(time_step_list[0], time_step_list[-1])
    plt.xticks(time_step_list)
    plt.gca().set_xlabel("Days", loc='center')
    plt.gca().set_ylim(0, 18)
    plt.gca().set_ylabel("NPV, MM$", loc='center')
    plt.grid()
    plt.gca().set_title(f"NPV (MM$, Step #{algorithm_iter_count}, Sample #{sample_num})", font="Arial", fontsize=16)
    figname = f'NPV-Step{algorithm_iter_count}-Sample{sample_num}' + '.png'
    plt.savefig(os.path.join(args.figure_directory, f'Step{algorithm_iter_count}-WP', 'NPV', figname))
    plt.close()

########## Visualization of NPV and Well Placement for all samples at each algorithm iteration count (Step) ##########
def _visualization_average(args, simulation_sample: list[WellPlacementSample], algorithm_iter_count: int) -> None:
    # # 1. Well placement visualization
    coord_x = []
    coord_y = []

    for sample_num in range(len(simulation_sample)):
        for well_num in range(args.total_well_num_max):
            coord_x.append(simulation_sample[sample_num].well_loc_list[well_num][0]-1) # 0 ~ args.gridnum_x
            coord_y.append(simulation_sample[sample_num].well_loc_list[well_num][1]-1) # 0 ~ args.gridnum_y
    plt.figure(figsize=(8,6)) # Unit: inch
    plt.tight_layout()
    plt.scatter(coord_x, coord_y, c='k')
    plt.imshow(np.log(np.array(args.perm_field)).reshape(15,15), cmap='jet')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("ln(Perm)")
    plt.gca().xaxis.tick_top()
    plt.gca().set_xticks(range(0, 15))
    plt.gca().set_xlabel("Grid X", loc='center')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_yticks(range(0, 15))
    plt.gca().set_ylabel("Grid Y")
    plt.gca().set_title(f"ln(Perm) map with all well location (Step #{algorithm_iter_count}, Total Sample Num. {args.sample_num_per_iter})", font="Arial", fontsize=16)
    figname = f'All Well placement-Step{algorithm_iter_count}' + '.png'
    plt.savefig(os.path.join(args.figure_directory, figname))
    plt.close()

    # # 2. NPV visualization
    npv_dict = {}
    avg_npv_dict = {}

    # Save all NPV information in simulation_sample
    for sample_num in range(len(simulation_sample)):
        npv_dict[f'Step{algorithm_iter_count}_Sample{sample_num+1}'] = [0] + list(np.cumsum(np.array(simulation_sample[sample_num].income)) * args.reward_unit)

    time_step_list = [args.time_step * i for i in range(args.total_well_num_max + 1)]

    # Graph of NPV of all samples
    plt.figure(figsize=(8, 6))  # Unit: inch
    for sample_num in range(len(simulation_sample)):
        plt.plot(time_step_list, np.array(npv_dict[f'Step{algorithm_iter_count}_Sample{sample_num+1}']) / (10 ** 6), color='silver')
    plt.gca().set_xlim(time_step_list[0], time_step_list[-1])
    plt.xticks(time_step_list)
    plt.gca().set_xlabel("Days", loc='center')
    plt.gca().set_ylim(0, 18)
    plt.gca().set_ylabel("NPV, MM$", loc='center')
    plt.grid()
    plt.gca().set_title(f"NPV Value (MM$, Step {algorithm_iter_count})", font="Arial", fontsize=16)

    # Graph of Average NPV for all samples
    npv_array = np.zeros(len(time_step_list))
    for sample_num in range(len(simulation_sample)):
        for tstep_idx in range(args.total_well_num_max): # Initial NPV == 0 for this case
            npv_array[tstep_idx+1] += npv_dict[f'Step{algorithm_iter_count}_Sample{sample_num+1}'][tstep_idx+1]
    # avg_npv_dict[f'Step{algorithm_iter_count}'] = list(npv_array / args.sample_num_per_iter)
    avg_npv_dict[f'Step{algorithm_iter_count}'] = list(npv_array / len(simulation_sample))
    plt.plot(time_step_list, np.array(avg_npv_dict[f'Step{algorithm_iter_count}']) / (10 ** 6), color='orange')
    plt.gca().legend(['NPV of each samples', 'Average NPV for all samples'], loc='upper left')

    ax = plt.gca()
    leg = ax.get_legend()
    leg.legend_handles[0].set_color('silver') # Legend: 'NPV of each samples'
    leg.legend_handles[1].set_color('orange') # Legend: 'Average NPV for all samples'

    figname = f'NPV-Step{algorithm_iter_count}' + '.png'
    plt.savefig(os.path.join(args.figure_directory, figname))
    plt.close()