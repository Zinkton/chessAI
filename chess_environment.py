import chess
import torch
import evaluation
import numpy
from evaluation import piece_value, position_value
from typing import List, Tuple

class ChessEnvironment:
    def __init__(self, fen = chess.Board.starting_fen):
        self.board = chess.Board(fen)
        self.position_score = 0

    def get_all_move_states_targets(self, state):
        max_value = float(evaluation.piece_value[chess.KING]) * 2.0
        states = []
        add_states = []
        targets = []

        for move in self.board.legal_moves:
            position_score = self.position_score
            move_value = self._calculate_move_value(move)
            
            position_score += move_value if self.board.turn else -move_value

            inverted_state = self._make_move_on_state(state, move)

            self.board.push(move)

            new_additional_state = self.get_additional_state()
            new_additional_state = self.invert_additional_state(new_additional_state)
            new_board_state = self.invert_state(state) if inverted_state is None else inverted_state

            states.append(new_board_state)
            add_states.append(new_additional_state)
            position_score /= max_value
            if not self.board.turn:
                position_score = -position_score
            targets.append(position_score)

            self.board.pop()

        return states, add_states, targets

    def print_8x8_tensor(self, tensor):
        """
        Prints an 8x8 space-separated representation of a 64-value torch float tensor.
        Assumes the tensor is 1D with 64 elements.
        """
        if tensor.numel() != 64:
            raise ValueError("Tensor must have exactly 64 elements.")

        # Reshape tensor to 8x8 for printing
        tensor_8x8 = tensor.view(8, 8)

        for row in tensor_8x8:
            print(' '.join(f'{value*6:.0f}' for value in row.tolist()))

    def reset(self):
        """
        Resets the chessboard to the initial state.
        """
        self.board.reset()
        self.position_score = 0
        return self.get_board_state(), self.get_additional_state()
    
    def reset_numpy(self):
        """
        Resets the chessboard to the initial state.
        """
        self.board.reset()
        self.position_score = 0
        return self.get_board_state_numpy(), self.get_additional_state_numpy()
    
    def _check_castling(self, move, board_state):
        if self.board.kings & chess.BB_SQUARES[move.from_square]:
            diff = chess.square_file(move.from_square) - chess.square_file(move.to_square)
            if abs(diff) > 1:
                is_king_castling = diff < 0
                k_index = 4
                k_target = None
                r_target = None
                r_index = None
                if is_king_castling:
                    r_index = 7
                    r_target = 5
                    k_target = 6
                else:
                    r_index = 0
                    r_target = 3
                    k_target = 2

                board_state[[k_index, k_target]] = board_state[[k_target, k_index]]
                board_state[[r_index, r_target]] = board_state[[r_target, r_index]]

                return True
            
        return False
    
    def _check_promotion(self, move: chess.Move, board_state):
        if move.promotion:
            board_state[move.from_square] = 0.0
            board_state[move.to_square] = move.promotion / 6.0 if self.board.turn else -move.promotion / 6.0
            
            return True
        
        return False
    
    def get_additional_state(self):
        # Get additional state information and convert to tensor
        ep = -1.0
        if self.board.ep_square is not None:
            ep = self.board.ep_square
            if 16 <= ep <= 23:
                ep = ep - 16
            elif 40 <= ep <= 47:
                ep = ep - 40
            ep = ep / 7.0
        w_k_castling = float(self.board.has_kingside_castling_rights(chess.WHITE))
        b_k_castling = float(self.board.has_kingside_castling_rights(chess.BLACK))
        w_q_castling = float(self.board.has_queenside_castling_rights(chess.WHITE))
        b_q_castling = float(self.board.has_queenside_castling_rights(chess.BLACK))
        additional_state = [ep, w_k_castling, b_k_castling, w_q_castling, b_q_castling, self.board.halfmove_clock / 100.0]

        return additional_state
    
    def get_additional_state_numpy(self):
        return numpy.array(self.get_additional_state(), dtype=numpy.float32)

    def _move_piece_on_state(self, board_state, move: chess.Move):
        if not self._check_promotion(move, board_state):
            board_state[move.to_square] = board_state[move.from_square]
            board_state[move.from_square] = 0.0

    def get_board_state(self):
        """
        Converts the board state to a numerical format that can be used as input to the neural network.
        """
        # Initialize a PyTorch tensor for the board state
        board_state = torch.zeros(64, dtype=torch.float32)

        # Populate the tensor based on the board
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                board_state[i] = piece.piece_type / 6.0 if piece.color == chess.WHITE else -piece.piece_type / 6.0

        return board_state
    
    def get_board_state_numpy(self):
        """
        Converts the board state to a numerical format that can be used as input to the neural network.
        """
        # Initialize a PyTorch tensor for the board state
        board_state = numpy.zeros(64, dtype=numpy.float32)

        # Populate the tensor based on the board
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                board_state[i] = piece.piece_type / 6.0 if piece.color == chess.WHITE else -piece.piece_type / 6.0

        return board_state
        
    def invert_state(self, state) -> torch.tensor:
        # Negate the tensor values
        inverted_state = -state

        # Reshape the tensor to 8x8, reverse each row, then flatten back to 1D
        inverted_state = inverted_state.view(8, 8).flip(dims=[0]).flatten()

        return inverted_state
    
    def invert_state_numpy(self, state):
        # Negate the array values
        inverted_state = -state

        # Reshape the array to 8x8
        inverted_state = inverted_state.reshape(8, 8)

        # Reverse each row
        inverted_state = numpy.flip(inverted_state, axis=0)

        # Flatten the array back to 1D
        inverted_state = inverted_state.flatten()

        return inverted_state
    
    def invert_additional_state(self, additional_state):
        # Invert castling rights
        w_k_castling = additional_state[1]
        b_k_castling = additional_state[2]
        w_q_castling = additional_state[3]
        b_q_castling = additional_state[4]

        inverted_additional_state = [additional_state[0], b_k_castling, w_k_castling, b_q_castling, w_q_castling, additional_state[5]]

        return inverted_additional_state
    
    def invert_additional_state_numpy(self, additional_state):
        # Invert castling rights
        additional_state[[1, 2, 3, 4]] = additional_state[[2, 1, 4, 3]]

        return additional_state

    def _make_move_on_state(self, state, move):
        inverted_state = None

        # Checking castling doesn't require inverting the board
        if not self._check_castling(move, state):
            # We need to invert the board before making a move if we're black
            if self.board.turn:
                self._move_piece_on_state(state, move)
            else:
                inverted_state = self.invert_state(state)
                self._move_piece_on_state(inverted_state, move)

        return inverted_state
    
    def _make_move_on_state_numpy(self, state, move):
        inverted_state = None

        # Checking castling doesn't require inverting the board
        if not self._check_castling(move, state):
            # We need to invert the board before making a move if we're black
            if self.board.turn:
                self._move_piece_on_state(state, move)
            else:
                inverted_state = self.invert_state_numpy(state)
                self._move_piece_on_state(inverted_state, move)

        return inverted_state
    
    def step(self, move, state) -> Tuple[torch.Tensor, List, chess.Outcome]:
        move_value = self._calculate_move_value(move)
        self.position_score += move_value if self.board.turn else -move_value

        inverted_state = self._make_move_on_state(state, move)

        self.board.push(move)
        outcome = self.board.outcome()

        # We return inverted position, perspective should always be as if the agent is white
        new_additional_state = self.get_additional_state()
        new_additional_state = self.invert_additional_state(new_additional_state)
        new_board_state = self.invert_state(state) if inverted_state is None else inverted_state
        
        return new_board_state, new_additional_state, outcome
    
    def step_numpy(self, move, state) -> Tuple[numpy.ndarray, List, chess.Outcome]:
        move_value = self._calculate_move_value(move)
        self.position_score += move_value if self.board.turn else -move_value

        inverted_state = self._make_move_on_state_numpy(state, move)

        self.board.push(move)
        outcome = self.board.outcome()

        # We return inverted position, perspective should always be as if the agent is white
        new_additional_state = self.get_additional_state_numpy()
        self.invert_additional_state_numpy(new_additional_state)
        new_board_state = self.invert_state_numpy(state) if inverted_state is None else inverted_state
        
        return new_board_state, new_additional_state, outcome
    
    def get_legal_moves(self):
        return self.board.legal_moves
    
    def get_game_result(self):
        outcome = self.board.outcome()
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return 1
            elif outcome.winner == chess.BLACK:
                return 0
        # draw
        return 0.5
    
    def get_outcome_score(self, outcome: chess.Outcome):        
        if outcome.termination == chess.Termination.CHECKMATE:
            return self.position_score + piece_value[chess.KING] if outcome.winner == chess.WHITE else -piece_value[chess.KING] - self.position_score
        
        return 0
    
    def _calculate_move_value(self, move):
        castle_value = self._check_castling_value(move)
        if castle_value is not None:
            return castle_value
    
        value = 0
        src_piece = self.board.piece_type_at(move.from_square)
        dest_piece = self.board.piece_type_at(move.to_square)
        
        if move.promotion:
            prom = move.promotion
            pos_score = position_value[self.board.turn][prom][move.to_square] - position_value[self.board.turn][src_piece][move.from_square]
            prom_score = piece_value[prom] - piece_value[src_piece]
            value = pos_score + prom_score
        else:
            value = position_value[self.board.turn][src_piece][move.to_square] - position_value[self.board.turn][src_piece][move.from_square]

        if dest_piece:
            # If we capture, add score based on captured piece value and position value
            value += piece_value[dest_piece] + position_value[not self.board.turn][dest_piece][move.to_square]
            
        return value

    def _check_castling_value(self, move):
        if self.board.kings & chess.BB_SQUARES[move.from_square]:
            diff = chess.square_file(move.from_square) - chess.square_file(move.to_square)
            if abs(diff) > 1:
                return evaluation.K_CASTLING_VALUE if diff < 0 else evaluation.Q_CASTLING_VALUE
            
        return None

    def render(self):
        """
        Prints the current board state. Useful for debugging.
        """
        print(self.board)
        
    