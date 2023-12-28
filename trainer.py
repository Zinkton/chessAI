from multiprocessing import Pool
import torch.optim as optim
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment
import time
from generate_training_data_input import GenerateTrainingDataInput
import model_utilities
import constants
import logging
import torch
import chess
import elo_calculator
from evaluation import piece_value

def generate_training_data_multiprocess(agent: ChessAgent, models, alpha):
    generate_training_data_input = [GenerateTrainingDataInput(agent, ChessEnvironment(), model_utilities.load_model(models), bool(x % 2), alpha) for x in range(constants.THREADS)]
        
    with Pool(processes=constants.THREADS) as p:
        generated_data = p.map(generate_training_data, generate_training_data_input)

    filtered_states = []
    filtered_add_states = []
    filtered_episode_targets = []

    draw_states = []
    draw_add_states = []
    draw_episode_targets = []

    for episode_states, episode_add_states, episode_targets, is_checkmate in generated_data:
        if is_checkmate:
            filtered_states.extend(episode_states)
            filtered_add_states.extend(episode_add_states)
            filtered_episode_targets.extend(episode_targets)
        else:
            draw_states.extend(episode_states)
            draw_add_states.extend(episode_add_states)
            draw_episode_targets.extend(episode_targets)

    # Only 1 draw will suffice, I want to prioritize data with checkmates
    if draw_states:
        filtered_states.append(draw_states[0])
        filtered_add_states.append(draw_add_states[0])
        filtered_episode_targets.append(draw_episode_targets[0])

    return filtered_states, filtered_add_states, filtered_episode_targets

def generate_training_data(input: GenerateTrainingDataInput):
    (agent, env, opponent, is_white, alpha) = (input.agent, input.env, input.opponent, input.is_white, input.alpha)
    episode_states = []
    episode_add_states = []
    episode_targets = []
    is_checkmate = False

    state_1, add_state_1 = env.reset()
    outcome = None
    
    # Opponent is white and makes first move
    if not is_white:
        _, state_1, add_state_1, outcome = model_utilities.make_move(env, opponent)
    
    while outcome is None:
        episode_state = None
        episode_add_state = None
        episode_target = None

        # Make a move and get a new state
        _, state_2, add_state_2, outcome = model_utilities.make_move(env, agent)

        # ANALYZE CAREFULLY WITH SOBER BRAIN
        if outcome is None:
            # Opponent makes move
            _, state_3, add_state_3, outcome = model_utilities.make_move(env, opponent)
            # Since opponent made a move, we reevaluate the position and update for position before enemy move state_2
            episode_state = state_2
            episode_add_state = add_state_2
            if outcome is None:
                # We want to evaluate position from opponent POV, so we invert
                state_3_inv, add_state_3_inv  = env.invert(state_3, add_state_3)
                evaluation = agent.get_position_evaluation(state_3_inv.unsqueeze(0), add_state_3_inv)
                # From opponent point of view, so score is opposite sign if agent is white
                position_score = -env.position_score if is_white else env.position_score
                position_score = position_score / float(piece_value[chess.KING])
                episode_target = alpha * position_score + (1 - alpha) * evaluation

                state_1 = state_3
                add_state_1 = add_state_3
            else:
                outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
                episode_target = -outcome_score if is_white else outcome_score
        else:
            outcome_score = env.get_outcome_score(outcome) / float(piece_value[chess.KING])
            # Since opponent did not make a move, we update for position before agent move state_1
            episode_state = state_1
            episode_add_state = add_state_1
            episode_target = outcome_score if is_white else -outcome_score
        
        episode_add_state = torch.FloatTensor(episode_add_state.copy()).to('cuda')
        episode_target = episode_target
        episode_target = torch.FloatTensor([episode_target]).to("cuda")

        episode_states.append(episode_state)
        episode_add_states.append(episode_add_state)
        episode_targets.append(episode_target)
    
    is_checkmate = outcome.termination == chess.Termination.CHECKMATE

    return episode_states, episode_add_states, episode_targets, is_checkmate

def do_training():
    logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG)
    logging.info('do_training Started')
    models = {}
    agent = model_utilities.load_latest_model()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    
    epoch = int(agent.name)
    while True:
        logging.info('Epoch', epoch, 'Started')
        epoch_start = time.perf_counter()
        epoch_steps = 0

        batch_states = []
        batch_add_states = []
        batch_targets = []

        for iteration in range(constants.ITERATIONS_PER_EPOCH):
            logging.info('Epoch', epoch, 'Iteration', iteration, 'started')
            start = time.perf_counter()
            iteration_steps = 0
            iteration_loss = 0

            while len(batch_states) < constants.BATCH_SIZE:
                states, add_states, targets = generate_training_data_multiprocess(agent, models, constants.ALPHA)

                # Accumulate the states and targets
                batch_states.extend(states)
                batch_add_states.extend(add_states)
                batch_targets.extend(targets)
            
            # Convert batch data to tensors
            batch_states_tensor = torch.stack(batch_states[:constants.BATCH_SIZE]).to('cuda')
            batch_add_states = torch.stack(batch_add_states[:constants.BATCH_SIZE]).to('cuda')
            batch_targets_tensor = torch.stack(batch_targets[:constants.BATCH_SIZE]).to('cuda')

            # Perform training step
            optimizer.zero_grad()
            outputs = agent(batch_states_tensor)
            loss = loss_function(outputs, batch_targets_tensor)
            loss.backward()
            optimizer.step()

            # Clear the batches for next iteration
            del batch_states[:constants.BATCH_SIZE]
            del batch_targets[:constants.BATCH_SIZE]

            # Logging and accumulating iteration statistics
            iteration_loss = loss.item()
            iteration_steps = len(batch_states_tensor)
            logging.info(f'Epoch {epoch}, Iteration {iteration}, Steps {iteration_steps}, Loss {iteration_loss}, Took {time.perf_counter() - start}')

            epoch_steps += iteration_steps
        
        # Save the model and log epoch statistics
        model_utilities.save_model(agent, epoch)
        elo_calculator.calculate_elo(agent, models)
        logging.info(f'Epoch {epoch}, Steps {epoch_steps}, Took {time.perf_counter() - epoch_start}')
        epoch += 1
        
        


do_training()