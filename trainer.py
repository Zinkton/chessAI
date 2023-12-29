import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment
import time
from generate_training_data_input import GenerateTrainingDataInput
import model_utilities
import constants
import logging
import torch
import chess
from elo_calculator import EloCalculator
from evaluation import piece_value
from multithreading import multithreading_pool

def generate_training_data_multiprocess(agent: ChessAgent, models, alpha):
    generate_training_data_inputs = []
    for x in range(constants.THREADS):
        agent_copy = model_utilities.copy_model(agent)
        env = ChessEnvironment()
        opponent = model_utilities.copy_model(model_utilities.load_model(models, agent.name)) if x > 5 else agent_copy
        is_white = bool(x % 2)
        generate_training_data_inputs.append(GenerateTrainingDataInput(agent_copy, env, opponent, is_white, alpha))

    
    generated_data = multithreading_pool(generate_training_data, generate_training_data_inputs)

    states = []
    add_states = []
    targets = []

    for episode_states, episode_add_states, episode_targets, is_checkmate in generated_data:
        states.extend(episode_states)
        add_states.extend(episode_add_states)
        targets.extend(episode_targets)

    raise Exception("stop here")

    return states, add_states, targets

def generate_training_data(input: GenerateTrainingDataInput):
    (agent, env, opponent, is_white, alpha) = (input.agent, input.env, input.opponent, input.is_white, input.alpha)
    self_play = agent is opponent
    episode_states = []
    episode_add_states = []
    episode_targets = []
    is_checkmate = False

    state_1, add_state_1 = env.reset()
    outcome = None
    
    # Opponent is white and makes first move
    if not is_white:
        _, state_1, add_state_1, outcome = model_utilities.make_move(env, opponent, state_1.clone(), is_deterministic = not self_play)
    
    while outcome is None:
        episode_state = None
        episode_add_state = None
        episode_target = None

        position_score = env.position_score if is_white else -env.position_score
        position_score = position_score / float(piece_value[chess.KING])

        # Make a move and get a new state
        state_2_eval, state_2, add_state_2, outcome = model_utilities.make_move(env, agent, state_1.clone())

        # ANALYZE CAREFULLY WITH SOBER BRAIN
        if outcome is None:
            if self_play:
                # We can use state_1 as training data now, because it's self play
                episode_states.append(state_1)
                self_play_add_state = add_state_1.copy()
                episode_add_states.append(self_play_add_state)
                # The evaluation of the position is just the evaluation after the move with opposite sign
                evaluation = state_2_eval[1] * -1.0
                self_play_target = alpha * position_score + (1.0 - alpha) * evaluation
                episode_targets.append(self_play_target)

            episode_state = state_2
            episode_add_state = add_state_2
            # From opponent point of view, so score is opposite sign if agent is white
            position_score = -env.position_score if is_white else env.position_score
            position_score = position_score / float(piece_value[chess.KING])
            # Opponent makes move
            state_3_eval, state_3, add_state_3, outcome = model_utilities.make_move(env, opponent, state_2.clone(), is_deterministic = not self_play)
            
            if outcome is None:
                evaluation = None
                # We want to evaluate position from opponent POV, so we invert
                if self_play:
                    evaluation = state_3_eval[1] * -1
                else:
                    state_3_inv, add_state_3_inv  = env.invert_state(state_3), env.invert_additional_state(add_state_3)
                    evaluation = agent.get_position_evaluation(state_3_inv.unsqueeze(0), add_state_3_inv)
                
                episode_target = alpha * position_score + (1.0 - alpha) * evaluation

                state_1 = state_3
                add_state_1 = add_state_3
            else:
                outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
                episode_target = -outcome_score if is_white else outcome_score

                # Also include the position of outcome
                outcome_state = state_3
                outcome_add_state = add_state_3.copy()
                outcome_target = -episode_target

                episode_states.append(outcome_state)
                episode_add_states.append(outcome_add_state)
                episode_targets.append(outcome_target)
        else:
            outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
            # Since opponent did not make a move, we update for position before agent move state_1
            episode_state = state_1
            episode_add_state = add_state_1
            episode_target = outcome_score if is_white else -outcome_score

            # Also include the position of outcome
            outcome_state = state_2
            outcome_add_state = add_state_2.copy()
            outcome_target = -episode_target

            episode_states.append(outcome_state)
            episode_add_states.append(outcome_add_state)
            episode_targets.append(outcome_target)
        
        episode_add_state = episode_add_state.copy()
        episode_target = episode_target

        episode_states.append(episode_state)
        episode_add_states.append(episode_add_state)
        episode_targets.append(episode_target)
    
    # A mark if the episode was ended with a checkmate, for possible data filtering in the future
    is_checkmate = outcome.termination == chess.Termination.CHECKMATE

    # We can double training data by inverting the states and changing the sign of the target
    # We also convert the data to tensors
    final_episode_states = []
    final_episode_add_states = []
    final_episode_targets = []
    for x in range(len(episode_states)):
        bonus_episode_state = env.invert_state(episode_states[x])
        bonus_episode_add_state = torch.tensor(env.invert_additional_state(episode_add_states[x]), dtype=torch.float32)
        bonus_episode_target = -episode_targets[x]

        final_episode_states.append(bonus_episode_state)
        final_episode_states.append(episode_states[x])

        final_episode_add_states.append(bonus_episode_add_state)
        final_episode_add_states.append(torch.tensor(episode_add_states[x], dtype=torch.float32))

        final_episode_targets.append(bonus_episode_target)
        final_episode_targets.append(episode_targets[x])

    return final_episode_states, final_episode_add_states, final_episode_targets, is_checkmate

def do_training():
    logging.basicConfig(filename='logs.log', level=logging.DEBUG)
    logging.info('do_training Started')
    elo_calculator = EloCalculator()
    models = {}
    agent = model_utilities.load_latest_model()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    
    epoch = int(agent.name)
    while True:
        logging.info(f'Epoch {epoch} Started')
        epoch_start = time.perf_counter()
        epoch_steps = 0
        epoch_loss = 0

        for iteration in range(constants.ITERATIONS_PER_EPOCH):
            batch_states = []
            batch_add_states = []
            batch_targets = []

            while len(batch_states) < constants.BATCH_SIZE:
                states, add_states, targets = generate_training_data_multiprocess(agent, models, constants.ALPHA)

                # Accumulate the states and targets
                batch_states.extend(states)
                batch_add_states.extend(add_states)
                batch_targets.extend(targets)
            
            # Convert batch data to tensors
            batch_states_tensor = torch.stack(batch_states[:constants.BATCH_SIZE]).to('cuda')
            batch_add_states_tensor = torch.stack(batch_add_states[:constants.BATCH_SIZE]).to('cuda')
            batch_targets_tensor = torch.tensor(batch_targets[:constants.BATCH_SIZE], dtype=torch.float32).to('cuda')

            # Create a TensorDataset
            dataset = TensorDataset(batch_states_tensor, batch_add_states_tensor, batch_targets_tensor)

            # Create a DataLoader
            data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, shuffle=True)

            for batch_states_tensor, batch_add_states_tensor, batch_targets_tensor in data_loader:
                # Perform training step
                optimizer.zero_grad()
                outputs = agent(batch_states_tensor, batch_add_states_tensor)
                loss = loss_function(outputs, batch_targets_tensor)
                loss.backward()
                optimizer.step()

                # Accumulating iteration statistics
                iteration_loss = loss.item()

                epoch_loss += iteration_loss
                epoch_steps += len(batch_states_tensor)
        
        
        logging.info(f'Epoch {epoch}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss {epoch_loss / constants.ITERATIONS_PER_EPOCH}')
        epoch += 1

        # Save the model and log epoch statistics
        print(f'calling save_model {agent.name} {epoch}')
        model_utilities.save_model(agent, epoch)
        elo_calculator.calculate_elo(agent, models)

def measure_performance():
    agent = model_utilities.load_latest_model()
    agent_copy = model_utilities.copy_model(agent)
    env = ChessEnvironment()
    opponent = model_utilities.copy_model(agent)
    alpha = constants.ALPHA
    input = GenerateTrainingDataInput(agent_copy, env, opponent, True, alpha)
    total_steps = 0
    generate_training_data_inputs = []
    
    for x in range(constants.THREADS):
        agent_copy = model_utilities.copy_model(agent)
        env = ChessEnvironment()
        opponent = model_utilities.copy_model(model_utilities.load_model({}, agent.name)) if x > 5 else agent_copy
        is_white = bool(x % 2)
        generate_training_data_inputs.append(GenerateTrainingDataInput(agent_copy, env, opponent, is_white, alpha))    
    
    start = time.perf_counter()
    for input in generate_training_data_inputs:
        results, _, _, _ = generate_training_data(input)
        total_steps += len(results)
    print(total_steps / (time.perf_counter() - start))

    start = time.perf_counter()
    generated_data = multithreading_pool(generate_training_data, generate_training_data_inputs)
    print(total_steps / (time.perf_counter() - start))



if __name__ == '__main__':
    torch.set_default_device('cpu')
    #do_training()
    measure_performance()
    # env = ChessEnvironment()
    # state, add_state = env.get_board_state(), env.get_additional_state()
    # env.render()
    # model_utilities.print_8x8_tensor(state)
    # agent = ChessAgent()
    # selected_move, state, _, _ = model_utilities.make_move(env, agent, state, True)
    # print(f'move {selected_move[0]}, from_square: {selected_move[0].from_square}, to_square: {selected_move[0].to_square}')
    # env.render()
    # model_utilities.print_8x8_tensor(state)
    # selected_move, state, _, _ = model_utilities.make_move(env, agent, state, True)
    # print(f'move {selected_move[0]}, from_square: {selected_move[0].from_square}, to_square: {selected_move[0].to_square}')
    # env.render()
    # model_utilities.print_8x8_tensor(state)
    