###################### 1. Import required modules and arguments from parsing.py #############################
import torch.backends.cudnn

from parsing import *
from objects import *
# https://stackoverflow.com/questions/31519815/import-symbols-starting-with-underscore
# Functions start with underscore("_") can NOT be imported with wildcard letter("*")
from functions import *
from functions import _simulation_sampler, _experience_sampler, _visualization_average

def main():
    # Control Randomness
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) # For Multi-GPU environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    ######################################## 2. Run algorithm ##############################################
    # Directory setting
    if not os.path.exists(args.simulation_directory):
        print('Simulation directory does not exist: Created Simulation directory\n')
        os.mkdir(args.simulation_directory)

    if not os.path.exists(args.variable_save_directory):
        print('Variable storage directory does not exist: Created Variable storage directory\n')
        os.mkdir(args.variable_save_directory)

    if not os.path.exists(args.deeplearningmodel_save_directory):
        print('Deep learning model storage directory does not exist: Created Deep learning model storage directory\n')
        os.mkdir(args.deeplearningmodel_save_directory)

    if not os.path.exists(args.figure_directory):
        print('Figure directory does not exist: Created Figure directory\n')
        os.mkdir(args.figure_directory)

    if not os.path.exists(args.log_directory):
        print('Training log directory does not exist: Created Training log directory\n')
        os.mkdir(args.log_directory)

    # 2023-07-17
    # For drawing figures
    # Read permeability map
    args.perm_field = []

    with open(os.path.join(args.basicfilepath, args.perm_filename + ".DATA"), 'r') as file_read:
        lines = file_read.readlines()
        for line_num in range(args.gridnum_x * args.gridnum_y):
            args.perm_field.append(float(lines[line_num + 1]))

    # Implementation of DQN Algorithm
    # Initialize Deep Q Network
    Deep_Q_Network = DQN(args=args, block=BasicBlock).to('cuda')

    # Experience sample queue or Replay memory (double-ended queue, deque)
    replay_memory = Experience_list(args=args)

    optimizer = optim.AdamW(Deep_Q_Network.parameters(), lr=args.learning_rate, amsgrad=True)

    args.policy_param = args.policy_param_start
    policy_param_string = "epsilon" if args.policy_type == "e-Greedy" else "tau" if args.policy_type == "Boltzmann" else "None"

    for m in range(1, args.max_iteration + 1):
        # Load CNN model if it exists and move to next step (m)
        if os.path.exists(f'{args.deeplearningmodel_save_directory}\\DQN_Step_{m}.model'):
            checkpoint = torch.load(f'{args.deeplearningmodel_save_directory}\\DQN_Step_{m}.model', map_location=args.device)
            Deep_Q_Network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 2023-10-06: Append each experience.pkls to replay_memory for accidently halted code
            with open(f'{args.variable_save_directory}\\Experience_sample\\Experience_sample_{m}.pkl', 'rb') as expsam:
                experience_sample = pickle.load(expsam)
            for i in range(0, len(experience_sample)):
                if len(replay_memory) == args.replay_memory_size:
                    replay_memory.exp_list.popleft()
                replay_memory.exp_list.append(experience_sample[i])
            # Decrease policy parameter
            args.policy_param = policy_param_function(args=args, algorithm_iter_count=m)
            # Tracking variation of policy parameter
            args.policy_param_tracker.append(args.policy_param)
            print(f"Policy parameter ({policy_param_string}) at step {m}: ", args.policy_param, "\n")
            continue

        # Initialization of log file for CNN training sequence log
        if os.path.exists(f"{args.log_directory}\\DQN_Training_Step{m}.log"):
            os.remove(f"{args.log_directory}\\DQN_Training_Step{m}.log")

        # Generate well placement simulation sample list, Length of list is "args.sample_num_per_iter"
        simulation_sample = []

        # Total num. of experience == args.sample_num_per_iter * (args.total_production_time / args.time_step)
        for i in range(1, args.sample_num_per_iter + 1):
            simulation_sample.append(_simulation_sampler(args=args, algorithm_iter_count=m, sample_num=i, network=Deep_Q_Network, policy=args.policy_type))

        # Draw Average NPV and all Well placement sample
        _visualization_average(args=args, simulation_sample=simulation_sample, algorithm_iter_count=m)

        # Save simulation samples
        if not os.path.exists(os.path.join(args.variable_save_directory, 'Simulation_sample')):
            os.mkdir(os.path.join(args.variable_save_directory, 'Simulation_sample'))
        with open(f'{args.variable_save_directory}\\Simulation_sample\\Simulation_sample_{m}.pkl', 'wb') as simsam:
            pickle.dump(simulation_sample, simsam)

        # Generate experience sample list
        experience_sample = _experience_sampler(args=args, simulation_sample_list=simulation_sample)

        # Save experience samples
        if not os.path.exists(os.path.join(args.variable_save_directory, 'Experience_sample')):
            os.mkdir(os.path.join(args.variable_save_directory, 'Experience_sample'))
        with open(f'{args.variable_save_directory}\\Experience_sample\\Experience_sample_{m}.pkl', 'wb') as expsam:
            pickle.dump(experience_sample, expsam)

        for i in range(0, len(experience_sample)):
            if len(replay_memory) == args.replay_memory_size:
                replay_memory.exp_list.popleft()
            replay_memory.exp_list.append(experience_sample[i])

        # 2023-09-11: Calclulate args.replay_batch_num with length of current experience list at current step (args.batch_size is fixed == 64)
        if m <= round(args.max_iteration * 0.3):
            args.replay_batch_num = ((len(replay_memory) // args.batch_size) + 1) * 2
            print(f"Current length of replay_memory: {len(replay_memory)}")
            print(f"Current args.replay_batch_num: {args.replay_batch_num}")
        else: # m >= round(args.max_iteration * 0.3) + 1
            args.replay_batch_num = ((round(args.max_iteration * args.sample_num_per_iter * args.total_well_num_max * 0.3) // args.batch_size) + 1) * 2
            print(f"Current length of replay_memory: {len(replay_memory)}")
            print(f"Current args.replay_batch_num: {args.replay_batch_num}")
        with open(f"{args.log_directory}\\DQN_Training_Step{m}.log", "a") as log_write:
            log_write.write(f"====== Info: Current Samplig Step {m} ======\n")
            log_write.write(f"Current length of replay_memory: {len(replay_memory)}\n")
            log_write.write(f"Current total batch num. at replay memory: {args.replay_batch_num}\n")
            log_write.write(f"Current policy parameter ({policy_param_string}): {args.policy_param}\n\n")

        for b in range(1, args.replay_batch_num + 1):
            # # Extract b-th experience data from replay memory
            target_Q = []  # Target Q value, yi # target_Q must be fixed!
            next_Q = []  # For calculation of Target Q value # next_Q must be fixed!

            # We want to know indices of Experience samples which are selected, but DataLoader does not support fuction
            # of searching indices of selected Experience samples directly.
            # Thus, (1) Select Experience samples in replay_memory and (2) Perform DQN training with selected subset of Experience samples
            ############################################################################################################
            # 2023-10-10: PER TEST
            if args.activate_PER == False:
                exp_idx = [random.randint(0, len(replay_memory)-1) for r in range(args.batch_size)] # Indices for subset
            # p = replay_memory.priority: deque, exp_dix = type(List)
            elif args.activate_PER == True:
                exp_priority = [np.power(replay_memory.exp_list[i].td_err + args.prob_compensation, args.prob_exponent).squeeze() for i in range(len(replay_memory))]
                exp_denominator = sum(exp_priority)
                exp_prob = [exp_priority[i]/exp_denominator for i in range(len(replay_memory))]
                exp_idx = np.random.choice(a=range(len(replay_memory)), size=args.batch_size, replace=False, p=exp_prob)
            subset_current = Experience_list(args=args)
            for i in range(args.batch_size):
                subset_current.exp_list.append(replay_memory.exp_list[exp_idx[i]])
            subset_next = deepcopy(subset_current) # For calculation of max. Q-value at next state
            for element in subset_next.exp_list: # Replace current state with next state, so if DataLoader called for subset_next, next state will be used for DQN.
                element.current_state = element.next_state
            batch_current = DataLoader(dataset=subset_current, batch_size=args.batch_size, shuffle=False)
            batch_next = DataLoader(dataset=subset_next, batch_size=args.batch_size, shuffle=False)
            ############################################################################################################

            for sample_current, sample_next in zip(batch_current, batch_next):
                # To get related data by searching replay_memory with exp_list, for loop was used.
                # Maximum Q-value at next_state for each Experience sample in batch only can be calculated as batch unit! (not Experience sample unit!)
                # Output dimension: (batch_size, 1, gridnum_y, gridnum_x)
                next_Q_map = Deep_Q_Network.forward(sample_next) # Tensor >> Tensor

                # Do Well placement masking before finding max. Q-value
                next_Q_map_mask = numpy.squeeze(deepcopy(next_Q_map.detach()).cpu().numpy(), axis=1)  # For 2-D well placement
                for i in range(args.batch_size):
                    for row in range(len(replay_memory.exp_list[exp_idx[i]].next_state[2])): # replay_memory.exp_list[exp_idx[i]].next_state[2]: Well placement map
                        for col in range(len(replay_memory.exp_list[exp_idx[i]].next_state[2][row])):
                            if replay_memory.exp_list[exp_idx[i]].next_state[2][row][col] == 1:
                                next_Q_map_mask[i][row][col] = np.NINF # (x, y) for ECL, (Row(y), Col(x)) for Python / 2D-map array

                for i in range(args.batch_size):
                    # # Q-value for current_state will always be used, but Q-value for next_state cannot be used if next_state is terminal state
                    # # Output dimension: (batch_size, 1, gridnum_y, gridnum_x)
                    # # (x, y) for ECL, (Row=y, Col=x) for Python [Index: 1~nx for ECL, 0~(nx-1) for Python] / 2D-map array
                    # current_Q.append(Deep_Q_Network.forward(sample_current)[i][0][int(subset_current.exp_list[i].current_action[1])-1][int(subset_current.exp_list[i].current_action[0])-1].reshape(1))
                    # max_action = max(Q at state s')
                    max_row, max_col = np.where(np.array(next_Q_map_mask[i]) == max(map(max, np.array(next_Q_map_mask[i]))))
                    # next_Q.append(next_Q_map_mask[i][max_row][max_col])
                    next_Q.append(next_Q_map_mask[i][max_row[0]][max_col[0]])

                    # if well_num == total_well_num_max (terminal state):
                    #   yi = ri
                    if np.cumsum(replay_memory.exp_list[exp_idx[i]].next_state[2].detach().cpu().numpy())[-1] == 5:  # sample.next_state[2]: Well placement map
                        target_Q.append(replay_memory.exp_list[exp_idx[i]].reward.reshape(1))

                    # elif well_num < total_well_num_max (non-terminal state):
                    #   yi = ri + args.discount_factor * max.Q_value(Q_network(s', a'))
                    elif np.cumsum(replay_memory.exp_list[exp_idx[i]].next_state[2].detach().cpu().numpy())[-1] < 5:  # sample.next_state[2]: Well placement map
                        target_Q.append((replay_memory.exp_list[exp_idx[i]].reward + args.discount_factor * (next_Q[i])).reshape(1))

                for u in range(1, args.nn_update_num + 1):
                    current_Q = []  # For calculation of loss
                    for i in range(args.batch_size):
                        current_Q.append(Deep_Q_Network.forward(sample_current)[i][0][int(subset_current.exp_list[i].current_action[1]) - 1][int(subset_current.exp_list[i].current_action[0]) - 1].reshape(1))

                    # # For debugging
                    targetQ_debug = []
                    currQ_debug = []
                    for i in range(len(current_Q)):
                        targetQ_debug.append(target_Q[i].detach().cpu().numpy()[0])
                        currQ_debug.append(current_Q[i].detach().cpu().numpy()[0])

                    # Loss calculation (Mean Square Error (MSE)): L(theta) = sum((yi - Q_network(s, a))^2) / args.batch_size
                    criterion = nn.MSELoss()
                    # criterion = nn.SmoothL1Loss()
                    loss = criterion(torch.cat(target_Q), torch.cat(current_Q))

                    # # For debugging (DQN training sequence log)
                    with open(f"{args.log_directory}\\DQN_Training_Step{m}.log", "a") as log_write:
                        log_write.write(f"====== Info: Step {m}, Batch #{b}, CNN Update #{u} ======\n")
                        log_write.write(f"Target Q Value = {targetQ_debug}\n")
                        log_write.write(f"Current Q Value = {currQ_debug}\n")
                        log_write.write(f"Loss = {loss.item()}\n\n")

                    ####################################################################################################
                    # 2023-10-10: PER TEST
                    # TD-error = Target_Q - Current_Q
                    if args.activate_PER == True:
                        for i in range(len(target_Q)): replay_memory.exp_list[exp_idx[i]].td_err = abs(target_Q[i].detach().cpu().numpy()[0] - current_Q[i].detach().cpu().numpy()[0])
                    ####################################################################################################

                    # Update Q-network parameter: theta = theta - args.learning_rate * grad(L(theta))
                    optimizer.zero_grad()
                    loss.requires_grad_(True)

                    loss.backward(retain_graph=True)
                    optimizer.step()

        # Decrease policy parameter
        args.policy_param = policy_param_function(args=args, algorithm_iter_count=m)
        # Tracking variation of policy parameter
        args.policy_param_tracker.append(args.policy_param)
        print(f"Policy parameter ({policy_param_string}) at step {m}: ", args.policy_param, "\n")

        # Save CNN model and optimizer
        torch.save({
            'model_state_dict': Deep_Q_Network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'{args.deeplearningmodel_save_directory}\\DQN_Step_{m}.model')

    # Do Last simulation sampling with Greedy policy
    args.policy_param = args.policy_param_end
    simulation_sample_last = []
    for i in range(1, 10 + 1):
        simulation_sample_last.append(_simulation_sampler(args=args, algorithm_iter_count=m+1, sample_num=i, network=Deep_Q_Network, policy='Greedy'))

    _visualization_average(args=args, simulation_sample=simulation_sample_last, algorithm_iter_count=m+1)

if __name__ == "__main__":
    main()
