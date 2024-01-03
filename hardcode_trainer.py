import chess
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
import validator
import pandas as pd
from multithreading import multithreading_pool
from evaluation import piece_value

def generate_training_data_multiprocess(agent: ChessAgent, chunk):
    generate_training_data_inputs = []
    for index, row in chunk.iterrows():
        moves = row['moves'].split(' ')
        generate_training_data_inputs.append(moves)
    
    # generated_data = multithreading_pool(generate_training_data, generate_training_data_inputs)
    generated_data = []
    for input in generate_training_data_inputs:
        generated_data.append(generate_training_data(input))

    states = []
    add_states = []
    targets = []

    for episode_states, episode_add_states, episode_targets in generated_data:
        states.extend(episode_states)
        add_states.extend(episode_add_states)
        targets.extend(episode_targets)

    return states, add_states, targets

def generate_training_data(moves):
    episode_states = []
    episode_add_states = []
    episode_targets = []
    env = ChessEnvironment()

    state, add_state = env.reset()
    episode_states.append(state.clone())
    episode_add_states.append(add_state.copy())
    episode_targets.append(env.position_score / float(piece_value[chess.KING]))
    
    for move_san in moves:
        move = env.board.parse_san(move_san)
        state, add_state, outcome = env.step(move, state)
        
        turn_sign = 1.0 if env.board.turn else -1.0

        if outcome is not None:
            episode_states.append(state.clone())
            episode_add_states.append(add_state.copy())
            episode_targets.append(env.get_outcome_score(outcome) * turn_sign / float(piece_value[chess.KING]))
            break
        else:
            episode_states.append(state.clone())
            episode_add_states.append(add_state.copy())
            episode_targets.append(env.position_score * turn_sign / float(piece_value[chess.KING]))

        # extra_states, extra_add_states, extra_targets = env.get_all_move_states_targets(state.clone())

        # episode_states.extend(extra_states)
        # episode_add_states.extend(extra_add_states)
        # episode_targets.extend(extra_targets)

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

    return final_episode_states, final_episode_add_states, final_episode_targets

def do_training():
    logging.basicConfig(filename='logs.log', level=logging.DEBUG)
    logging.info('do_training Started')
    
    agent = model_utilities.load_latest_model()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()

    epoch = int(agent.name)
    chunk_iterable = pd.read_csv('games.csv', chunksize=(constants.THREADS * 100), skiprows=range(1, 101 + epoch * constants.THREADS * 100))
    for chunk in chunk_iterable:
        logging.info(f'Epoch {epoch} Started')
        epoch_start = time.perf_counter()
        epoch_steps = 0
        epoch_loss = 0

        batch_states = []
        batch_add_states = []
        batch_targets = []

        iteration_loss = 0
        iteration_steps = 0
        for start_row in range(0, len(chunk), 8):
            small_chunk = chunk.iloc[start_row:start_row + 8]
            states, add_states, targets = generate_training_data_multiprocess(agent, small_chunk)

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
        data_loader = DataLoader(dataset, batch_size=500, shuffle=True)

        batch_count = 0
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
            batch_count += 1

        epoch_loss += iteration_loss
        epoch_steps += iteration_steps
        
        logging.info(f'Epoch {epoch}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss { epoch_loss / batch_count }')
        epoch += 1
        
        model_utilities.save_model(agent, epoch)
        agent_copy = model_utilities.copy_model(agent)
        print(validator.validate_scores(agent_copy))
        print(f'saved model and calculated stats {agent.name}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss { epoch_loss / batch_count }')

if __name__ == '__main__':
    torch.set_default_device('cpu')
    do_training()