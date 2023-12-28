import torch
import random
import os
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment

def copy_model(model):
    # Create a new instance
    new_model = ChessAgent()
    new_model.eval()
    # Copy the weights from the original model to the new one
    new_model.load_state_dict(model.state_dict())

    return new_model

def make_move(env: ChessEnvironment, model, is_deterministic = False):
    moves_evaluations = evaluate_moves(env, model)

    selected_move_evaluation = None
    if is_deterministic:
        selected_move_evaluation = min(moves_evaluations, key=lambda x: x[1])
    else:
        selected_move_evaluation = softmax_selection(moves_evaluations)
    state, additional_state, outcome = env.step(selected_move_evaluation[0])

    return selected_move_evaluation, state, additional_state, outcome

def print_8x8_tensor(tensor):
    """
    Prints an 8x8 space-separated representation of a 64-value torch float tensor.
    Assumes the tensor is 1D with 64 elements.
    """
    if tensor.numel() != 64:
        raise ValueError("Tensor must have exactly 64 elements.")

    # Reshape tensor to 8x8 for printing
    tensor_8x8 = tensor.view(8, 8)

    for row in tensor_8x8:
        print(' '.join(f'{value:.0f}' for value in row.tolist()))

def evaluate_moves(env: ChessEnvironment, model: ChessAgent):
    moves_evaluations = []
    states, add_states = [], []
    for move in env.board.legal_moves:
        env.board.push(move)
        board_state, additional_state = env.get_board_state()
        board_state, additional_state = env.invert(board_state, additional_state)
        
        additional_state_tensor = torch.tensor(additional_state.copy(), dtype=torch.float32)
        
        states.append(board_state)
        add_states.append(additional_state_tensor)
        moves_evaluations.append([move, 0, board_state, additional_state])
        env.board.pop()

    evaluations = model.get_multiple_position_evaluations(states, add_states)
    for x in range(len(evaluations)):
        moves_evaluations[x][1] = evaluations[x]

    return moves_evaluations

def softmax_selection(moves_evaluations):
    evaluations = torch.tensor([item[1] for item in moves_evaluations], dtype=torch.float32)

    # We want the moves with the lowest scores because it's from the opponent POV
    evaluations = evaluations.max() - evaluations

    probabilities = torch.softmax(evaluations, dim=0)

    move_indices = list(range(len(moves_evaluations)))
    selected_index = random.choices(move_indices, weights=probabilities.cpu(), k=1)[0]

    return moves_evaluations[selected_index]

def pick_random_opponent(filenames):
    opponents = sorted([int(filename) for filename in filenames])
    num_opponents = len(opponents)

    # Generate weights based on a quadratic distribution
    # The latest model (last in the list) gets the highest weight
    weights = [(i / num_opponents) ** 2 for i in range(num_opponents)]

    # Normalize weights so that the sum equals 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Ensure the newest model has a weight of 0.5 (50%)
    normalized_weights[-1] = 0.5
    remaining_weight = 1 - 0.5
    for i in range(num_opponents - 1):
        normalized_weights[i] *= remaining_weight

    # Select an opponent based on the weights
    selected_opponent = random.choices(opponents, weights=normalized_weights, k=1)[0]
    return str(selected_opponent)

def save_model(model, version):
    filename = f"models/{version}"
    model.name = str(version)
    torch.save(model.state_dict(), filename)

def load_latest_model():
    model = ChessAgent().to('cuda')

    files = os.listdir('models/.')
    if files:
        latest_model_file = str(max([int(filename) for filename in files]))
        model.load_state_dict(torch.load(f'models/{latest_model_file}'))
        model.name = latest_model_file
    else:
        model.name = '0'
        save_model(model, 0)

    return model

def load_model(models, filename = None, agent_name = None):
    model = ChessAgent()

    if filename is None:
        files = os.listdir('models/.')
        files = [file for file in files if str(file) != agent_name]
        if not files:
            model.name = '0'
            models['0'] = model
            model.eval()

            return model
        
        if len(files) == 1:
            filename = files[0]
        else:
            # Pick random opponent, favoring latest models
            filename = pick_random_opponent(files)
        
        if filename in models:
            return models[filename]
    
    model.load_state_dict(torch.load(f'models/{filename}'))
    model.name = filename
    models[filename] = model
    model.eval()

    return model

def load_ratings():
    ratings = {}
    if not os.path.exists('elos.txt'):
        return ratings
    
    with open('elos.txt', 'r+') as file:
        for line in file:
            name, elo = line.strip().split(' ')
            elo = float(elo)
            ratings[name] = elo

    return ratings

def save_ratings(elos):
    with open('elos.txt', 'w') as file:
        for key in elos:
            file.write(f"{key} {elos[key]}\n")

def play_game(input):
    agent: ChessAgent = input[0]
    opponent: ChessAgent = input[1]
    env: ChessEnvironment = input[2]
    is_switch = input[3]

    env.reset()

    if is_switch:
        bot1 = opponent
        bot2 = agent
    else:
        bot1 = agent
        bot2 = opponent

    outcome = None
    while outcome is None:
        _, _, _, outcome  = make_move(env, bot1, True)
        if not outcome:
            _, _, _, outcome  = make_move(env, bot2, True)

    # (1 if bot1 wins, 0 if bot2 wins, 0.5 for a draw)
    game_result = env.get_game_result()
    if is_switch:
        game_result = 1 - game_result

    return game_result