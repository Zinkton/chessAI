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
import validator
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

    
    # ~3 times slower if it's 2 processes, ~4 times slower if it's 1 process
    generated_data = multithreading_pool(generate_training_data, generate_training_data_inputs)

    # This is commented code for the non multithreading option, it's as fast as ~ 8 threads with Pool
    # generated_data = []
    # for input in generate_training_data_inputs:
    #     generated_data.append(generate_training_data(input))

    states = []
    add_states = []
    targets = []

    for episode_states, episode_add_states, episode_targets, is_checkmate in generated_data:
        states.extend(episode_states)
        add_states.extend(episode_add_states)
        targets.extend(episode_targets)

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
        _, state_1, add_state_1, outcome = model_utilities.make_move(env, opponent, state_1.clone(), is_deterministic = True)
    
    while outcome is None:
        episode_state = None
        episode_add_state = None
        episode_target = None

        position_score = env.position_score if is_white else -env.position_score
        position_score = position_score / float(piece_value[chess.KING])

        # Make a move and get a new state
        state_2_eval, state_2, add_state_2, outcome = model_utilities.make_move(env, agent, state_1.clone(), is_deterministic = False)

        # ANALYZE CAREFULLY WITH SOBER BRAIN
        if outcome is None:
            if self_play and state_2_eval[4]:
                # We can use state_1 as training data now, because it's self play
                episode_states.append(state_1.clone())
                episode_add_states.append(add_state_1.copy())
                # The evaluation of the position is just the evaluation after the move with opposite sign
                evaluation = state_2_eval[1] * -1.0
                self_play_target = alpha * position_score + (1.0 - alpha) * evaluation
                episode_targets.append(self_play_target)

            episode_state = state_2.clone()
            episode_add_state = add_state_2.copy()
            # From opponent point of view, so score is opposite sign if agent is white
            position_score = -env.position_score if is_white else env.position_score
            position_score = position_score / float(piece_value[chess.KING])
            # Opponent makes move
            state_3_eval, state_3, add_state_3, outcome = model_utilities.make_move(env, opponent, state_2.clone(), is_deterministic = True)
            
            if outcome is None:
                evaluation = None
                # We want to evaluate position from opponent POV, so we invert
                if self_play:
                    evaluation = state_3_eval[1] * -1
                else:
                    state_3_inv, add_state_3_inv  = env.invert_state(state_3), env.invert_additional_state(add_state_3)
                    evaluation = agent.get_position_evaluation(state_3_inv.unsqueeze(0), add_state_3_inv)
                
                episode_target = alpha * position_score + (1.0 - alpha) * evaluation

                state_1 = state_3.clone()
                add_state_1 = add_state_3.copy()
            else:
                outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
                episode_target = -outcome_score if is_white else outcome_score

                # Also include the position of outcome
                outcome_state = state_3.clone()
                outcome_add_state = add_state_3.copy()
                outcome_target = -episode_target

                episode_states.append(outcome_state)
                episode_add_states.append(outcome_add_state)
                episode_targets.append(outcome_target)
        else:
            outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
            # Since opponent did not make a move, we update for position before agent move state_1
            episode_state = state_1.clone()
            episode_add_state = add_state_1.copy()
            episode_target = outcome_score if is_white else -outcome_score

            # Also include the position of outcome
            outcome_state = state_2.clone()
            outcome_add_state = add_state_2.copy()
            outcome_target = -episode_target

            episode_states.append(outcome_state)
            episode_add_states.append(outcome_add_state)
            episode_targets.append(outcome_target)
        
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
        bonus_episode_state = env.invert_state(episode_states[x].clone())
        bonus_episode_add_state = torch.tensor(env.invert_additional_state(episode_add_states[x]), dtype=torch.float32)
        bonus_episode_target = -episode_targets[x]

        final_episode_states.append(bonus_episode_state)
        final_episode_states.append(episode_states[x])

        final_episode_add_states.append(bonus_episode_add_state)
        final_episode_add_states.append(torch.tensor(episode_add_states[x], dtype=torch.float32))

        final_episode_targets.append(torch.tensor([bonus_episode_target], dtype=torch.float32))
        final_episode_targets.append(torch.tensor([episode_targets[x]], dtype=torch.float32))

    return final_episode_states, final_episode_add_states, final_episode_targets, is_checkmate

def do_training():
    logging.basicConfig(filename='logs.log', level=logging.DEBUG)
    logging.info('do_training Started')
    elo_calculator = EloCalculator()
    models = {}
    agent = model_utilities.load_latest_model()
    optimizer = optim.Adam(agent.parameters(), lr=constants.LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    epoch = int(agent.name)
    while True:
        alpha = 1.0 #max(1.0 - ((epoch + 1) / 100.0), 0)
        logging.info(f'Epoch {epoch} Started, alpha: {alpha}')
        epoch_start = time.perf_counter()
        epoch_steps = 0
        epoch_loss = 0

        for iteration in range(constants.ITERATIONS_PER_EPOCH):
            batch_states = []
            batch_add_states = []
            batch_targets = []

            iteration_loss = 0
            iteration_steps = 0
            iteration_start = time.perf_counter()
            while len(batch_states) < constants.BATCH_SIZE:
                states, add_states, targets = generate_training_data_multiprocess(agent, models, alpha)
                # states, add_states, targets = generate_training_data_multiprocess(agent, models, alpha)

                # Accumulate the states and targets
                batch_states.extend(states)
                batch_add_states.extend(add_states)
                batch_targets.extend(targets)
            
            # Convert batch data to tensors
            batch_states_tensor = torch.stack(batch_states).to('cuda')
            batch_add_states_tensor = torch.stack(batch_add_states).to('cuda')
            batch_targets_tensor = torch.stack(batch_targets).to('cuda')

            # Create a TensorDataset
            dataset = TensorDataset(batch_states_tensor, batch_add_states_tensor, batch_targets_tensor)

            # Create a DataLoader
            data_loader = DataLoader(dataset, batch_size=constants.MINIBATCH_SIZE, shuffle=True)

            for minibatch_batch_states_tensor, minibatch_batch_add_states_tensor, minibatch_batch_targets_tensor in data_loader:
                # Perform training step
                optimizer.zero_grad()
                outputs = agent(minibatch_batch_states_tensor, minibatch_batch_add_states_tensor)
                loss = loss_function(outputs, minibatch_batch_targets_tensor)
                loss.backward()
                optimizer.step()

                # Accumulating iteration statistics
                batch_loss = loss.item()
                iteration_loss += batch_loss

                iteration_steps += len(minibatch_batch_states_tensor)

            epoch_loss += iteration_loss
            epoch_steps += iteration_steps
            print(f'Iteration {iteration}, Steps per second: {iteration_steps / (time.perf_counter() - iteration_start)}, Loss {iteration_loss}')
        
        logging.info(f'Epoch {epoch}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss {epoch_loss / constants.ITERATIONS_PER_EPOCH}')
        epoch += 1
        
        model_utilities.save_model(agent, epoch)
        agent_copy = model_utilities.copy_model(agent)
        elo_calculator.calculate_elo(agent_copy, models)
        validator.validate_scores(agent_copy)
        print(f'saved model and calculated stats {agent.name}')


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
    total_data = []
    for item, _, _, _ in generated_data:
        total_data.extend(item)
    print(len(total_data) / (time.perf_counter() - start))

if __name__ == '__main__':
    torch.set_default_device('cpu')
    do_training()
    #do_training_hardcoded()
    # measure_performance()
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
    