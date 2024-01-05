import chess
import numpy
import random

from chess_environment import ChessEnvironment
from evaluation import piece_value

def generate_training_data_single_process(input):
    games, checkmate_chance = input[0], input[1]
    generated_data = []
    for moves in games:
        is_checkmate_only = False
        if checkmate_chance is not None:
            is_checkmate_only = checkmate_chance > random.random()
        generated_data.append(generate_training_data(moves, is_checkmate_only))

    states = []
    add_states = []
    targets = []

    for episode_states, episode_add_states, episode_targets in generated_data:
        states.extend(episode_states)
        add_states.extend(episode_add_states)
        targets.extend(episode_targets)

    return states, add_states, targets

def generate_training_data(moves, is_checkmate_only):
    episode_states = []
    episode_add_states = []
    episode_targets = []
    max_value = float(piece_value[chess.KING]) * 2.0
    env = ChessEnvironment()

    state, add_state = env.reset_numpy()
    if not is_checkmate_only:
        episode_states.append(state.copy())
        episode_add_states.append(add_state.copy())
        episode_targets.append(numpy.array([env.position_score / max_value], dtype=numpy.float32))

    for move in moves:
        state, add_state, outcome = env.step_numpy(move, state)
        
        turn_sign = 1.0 if env.board.turn else -1.0

        if outcome is not None:
            if is_checkmate_only:
                if outcome.termination == chess.Termination.CHECKMATE:
                    episode_states.append(state.copy())
                    episode_add_states.append(add_state.copy())
                    episode_targets.append(numpy.array([env.get_outcome_score(outcome) * turn_sign / max_value], dtype=numpy.float32))
            else:
                episode_states.append(state.copy())
                episode_add_states.append(add_state.copy())
                episode_targets.append(numpy.array([env.get_outcome_score(outcome) * turn_sign / max_value], dtype=numpy.float32))
            break
        elif not is_checkmate_only:
            episode_states.append(state.copy())
            episode_add_states.append(add_state.copy())
            episode_targets.append(numpy.array([env.position_score * turn_sign / max_value], dtype=numpy.float32))

        # extra_states, extra_add_states, extra_targets = env.get_all_move_states_targets(state.clone())

        # episode_states.extend(extra_states)
        # episode_add_states.extend(extra_add_states)
        # episode_targets.extend(extra_targets)

    # We can double training data by inverting the states and changing the sign of the target
    # We also convert the data to tensors
    for x in range(len(episode_states)):
        bonus_episode_state = env.invert_state_numpy(episode_states[x])
        bonus_episode_add_state = env.invert_additional_state_numpy(episode_add_states[x].copy())
        bonus_episode_target = -episode_targets[x]

        episode_states.append(bonus_episode_state)
        episode_add_states.append(bonus_episode_add_state)
        episode_targets.append(bonus_episode_target)

    return episode_states, episode_add_states, episode_targets