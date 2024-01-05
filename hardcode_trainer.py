import chess
import chess.pgn
import numpy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment
from hardcode_data_generator import generate_training_data_single_process
import model_utilities
import constants
import logging
import torch
import validator
from multithreading import multithreading_pool

def generate_training_data_multiprocess(games, games_per_thread):
    generate_training_data_inputs = []
    for x in range(constants.THREADS):
        generate_training_data_input = games[x * games_per_thread:x * games_per_thread + games_per_thread]
        generate_training_data_inputs.append([generate_training_data_input.copy(), 0.5])
    
    generated_data = multithreading_pool(generate_training_data_single_process, generate_training_data_inputs)

    states, add_states, targets = [], [], []
    for generated_states, generated_add_states, generated_targets in generated_data:
        states.extend(generated_states)
        add_states.extend(generated_add_states)
        targets.extend(generated_targets)

    return states, add_states, targets

def do_training():
    logging.basicConfig(filename='logs.log', level=logging.DEBUG)
    logging.info('do_training Started')
    
    agent = model_utilities.load_latest_model()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()

    epoch = int(agent.name)
    games_per_thread = 10000
    start_game = epoch * constants.THREADS * games_per_thread + 1
    current_game = 0
    total_avg_loss = 0
    with open('lichess_db_standard_rated_2014-10.pgn') as pgn_file:
        # skip the games that are already processed
        while current_game < start_game:
            chess.pgn.read_game(pgn_file)
            current_game += 1

        game = False
        while game is not None:
            logging.info(f'Epoch {epoch} Started')
            epoch_start = time.perf_counter()
            epoch_steps = 0
            epoch_loss = 0

            iteration_loss = 0
            iteration_steps = 0
            games = []
            for x in range(constants.THREADS * games_per_thread):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                moves = list(game.mainline_moves())
                games.append(moves)

            states, add_states, targets = generate_training_data_multiprocess(games, games_per_thread)

            # Convert batch data to tensors
            states = numpy.stack(states)
            add_states = numpy.stack(add_states)
            targets = numpy.stack(targets)
            states_tensor = torch.from_numpy(states)
            add_states_tensor = torch.from_numpy(add_states)
            targets_tensor = torch.from_numpy(targets)

            # Create a TensorDataset
            dataset = TensorDataset(states_tensor, add_states_tensor, targets_tensor)

            # Create a DataLoader
            data_loader = DataLoader(dataset, batch_size=64000, shuffle=True, num_workers=constants.THREADS)

            batch_count = 0
            
            for batch_states_tensor, batch_add_states_tensor, batch_targets_tensor in data_loader:
                batch_states_tensor_cuda = batch_states_tensor.cuda()
                add_states_tensor_cuda = batch_add_states_tensor.cuda()
                targets_tensor_cuda = batch_targets_tensor.cuda()
                # Perform training step
                optimizer.zero_grad()
                outputs = agent(batch_states_tensor_cuda, add_states_tensor_cuda)
                loss = loss_function(outputs, targets_tensor_cuda)
                loss.backward()
                optimizer.step()

                # Accumulating iteration statistics
                batch_loss = loss.item()
                iteration_loss += batch_loss

                iteration_steps += len(batch_states_tensor)
                batch_count += 1

            epoch_loss += iteration_loss
            epoch_steps += iteration_steps

            avg_loss = epoch_loss / batch_count
            total_avg_loss += avg_loss
            
            logging.info(f'Epoch {epoch}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss { avg_loss }')
            epoch += 1
            
            model_utilities.save_model(agent, epoch)
            agent_copy = model_utilities.copy_model(agent)
            print(validator.validate_scores(agent_copy))
            print(f'saved model and calculated stats {agent.name}, Steps per second: {epoch_steps / (time.perf_counter() - epoch_start) }, Loss { avg_loss }')

    run_avg_loss = total_avg_loss / epoch
    print(run_avg_loss)

def measure_get_training_data(games, games_per_thread):
    generate_training_data_inputs = []
    for x in range(constants.THREADS):
        generate_training_data_input = games[x * games_per_thread:x * games_per_thread + games_per_thread]
        generate_training_data_inputs.append(generate_training_data_input.copy())
    
    start = time.perf_counter()
    generated_data = []
    for input in generate_training_data_inputs:
        generated_data.append(generate_training_data_single_process(input))
    print(time.perf_counter() - start)
    
    start = time.perf_counter()
    generated_data = multithreading_pool(generate_training_data_single_process, generate_training_data_inputs)
    print(time.perf_counter() - start)

def measure_performance():
    with open('lichess_db_standard_rated_2013-01.pgn') as pgn_file:
        games = []
        games_per_thread = 1000
        for x in range(constants.THREADS * games_per_thread):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = list(game.mainline_moves())
            games.append(moves)

        measure_get_training_data(games, games_per_thread)

def test():
    agent = model_utilities.load_latest_model()
    agent = agent.cpu()
    agent.eval()
    env = ChessEnvironment()
    state, add_state = env.reset()
    with open('lichess_db_standard_rated_2013-01.pgn') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        moves = list(game.mainline_moves())
        print_evaluations(env, agent, state, add_state, None)
        for move in moves:
            state, add_state, outcome  = env.step(move, state)
            print_evaluations(env, agent, state, add_state, outcome)

def print_evaluations(env: ChessEnvironment, agent: ChessAgent, state, add_state, outcome):
    print(env.board)
    pos_score = env.position_score
    if outcome is not None:
        pos_score = env.get_outcome_score(outcome)
    evaluation1 = agent.get_position_evaluation(state, add_state)
    inv_state, inv_add_state = env.invert_state(state), env.invert_additional_state(add_state)
    evaluation2 = agent.get_position_evaluation(inv_state, inv_add_state)
    if not env.board.turn:
        print(pos_score, -evaluation1 * 8000.0, evaluation2 * 8000.0)
    else:
        print(pos_score, evaluation1 * 8000.0, -evaluation2 * 8000.0)
    

if __name__ == '__main__':
    torch.set_default_device('cpu')
    do_training()
    # measure_performance()

    # test()