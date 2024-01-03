import os
import pandas as pd
import torch
import chess
import model_utilities
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment
from evaluation import piece_value

def validate_scores(model: ChessAgent):
    validations = model_utilities.load_values('validations.txt')
    env = ChessEnvironment()
    df = pd.read_csv('games.csv', nrows=100)
    
    difference = 0.0
    for index, row in df.iterrows():
        state, add_state = env.reset()
        evaluations = []
        states, add_states = [], []
        states.append(state.clone())
        add_state_tensor = torch.tensor(add_state.copy(), dtype=torch.float32)
        add_states.append(add_state_tensor)
        turn_sign = 1.0 if env.board.turn else -1.0
        evaluations.append([env.position_score, turn_sign])
        moves = row['moves'].split(' ')
        for move_san in moves:
            move = env.board.parse_san(move_san)
            state, add_state, outcome = env.step(move, state)
            add_state_tensor = torch.tensor(add_state.copy(), dtype=torch.float32)
            turn_sign = 1.0 if env.board.turn else -1.0
            if outcome is not None:
                evaluations.append([env.get_outcome_score(outcome), turn_sign])
            else:
                evaluations.append([env.position_score, turn_sign])
            states.append(state.clone())
            add_states.append(add_state_tensor)
        
        model_evaluations = model.get_multiple_position_evaluations(states, add_states)
        for x in range(len(model_evaluations)):
            model_eval = model_evaluations[x][0] * evaluations[x][1]
            hardcoded_eval = evaluations[x][0] / float(piece_value[chess.KING])
            difference += abs(hardcoded_eval - model_eval)

    validations[model.name] = difference

    model_utilities.save_values(validations, 'validations.txt')

    return difference

if __name__ == '__main__':
    files = os.listdir('models/.')
    models = {}
    for file in files:
        model = model_utilities.load_model(models, file)
        
        validate_scores(model)
    