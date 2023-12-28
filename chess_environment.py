import chess
import torch
import evaluation
from evaluation import piece_value, position_value

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """
        Resets the chessboard to the initial state.
        """
        self.board.reset()
        self.position_score = 0
        return self.get_board_state()

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
                piece_value = {'P': 1/6.0, 'N': 2/6.0, 'B': 3/6.0, 'R': 4/6.0, 'Q': 5/6.0, 'K': 6/6.0,
                            'p': -1/6.0, 'n': -2/6.0, 'b': -3/6.0, 'r': -4/6.0, 'q': -5/6.0, 'k': -6/6.0}[piece.symbol()]
                board_state[i] = piece_value

        # Get additional state information and convert to tensor
        ep = -1
        if self.board.ep_square is not None:
            ep = self.board.ep_square
            if 16 <= ep <= 23:
                ep = ep - 16
            elif 40 <= ep <= 47:
                ep = ep - 40
            ep = ep / 7.0
        w_k_castling = int(self.board.has_kingside_castling_rights(chess.WHITE))
        b_k_castling = int(self.board.has_kingside_castling_rights(chess.BLACK))
        w_q_castling = int(self.board.has_queenside_castling_rights(chess.WHITE))
        b_q_castling = int(self.board.has_queenside_castling_rights(chess.BLACK))
        additional_state = [ep, w_k_castling, b_k_castling, w_q_castling, b_q_castling, self.board.halfmove_clock / 100.0]

        return board_state, additional_state
        
    def invert(self, state, additional_state):
        # Negate the tensor values
        state = -state

        # Reshape the tensor to 8x8, reverse each row, then flatten back to 1D
        state = state.view(8, 8).flip(dims=[0]).flatten()

        # Invert castling rights
        w_k_castling = additional_state[1]
        b_k_castling = additional_state[2]
        w_q_castling = additional_state[3]
        b_q_castling = additional_state[4]

        inverted_additional_state = [additional_state[0], b_k_castling, w_k_castling, b_q_castling, w_q_castling, additional_state[5]]

        return state, inverted_additional_state
    
    def step(self, move):
        move_value = self._calculate_move_value(move)
        self.position_score += move_value if self.board.turn else -move_value
        self.board.push(move)
        outcome = self.board.outcome(claim_draw = True)
        new_board_state, new_additional_state = self.get_board_state()
        new_board_state, new_additional_state = self.invert(new_board_state, new_additional_state)
        
        return new_board_state, new_additional_state, outcome
    
    def get_legal_moves(self):
        return self.board.legal_moves
    
    def get_game_result(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return 1
            elif outcome.winner == chess.BLACK:
                return 0
        # draw
        return 0.5
    
    def get_outcome_score(self, outcome: chess.Outcome):        
        if outcome.termination == chess.Termination.CHECKMATE:
            return piece_value[chess.KING] if outcome.winner == chess.WHITE else -piece_value[chess.KING]
        
        return 0
    
    def _calculate_move_value(self, move):
        castle_value = self._check_castling(self.board, move)
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

    def _check_castling(self, board, move):
        if board.kings & chess.BB_SQUARES[move.from_square]:
            diff = chess.square_file(move.from_square) - chess.square_file(move.to_square)
            if abs(diff) > 1:
                return evaluation.K_CASTLING_VALUE if diff < 0 else evaluation.Q_CASTLING_VALUE
            
        return None

    def render(self):
        """
        Prints the current board state. Useful for debugging.
        """
        print(self.board)
        
    