import chess
import numpy as np

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
        self.all_moves, self.all_moves_list = self.read_all_moves()
        self.piece_weights = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 0,
                                'p': -100, 'n': -280, 'b': -320, 'r': -479, 'q': -929, 'k': 0}
        self.pst = {
            'p': [
                     100, 100, 100, 100, 105, 100, 100,  100,
                      78,  83,  86,  73, 102,  82,  85,  90,
                       7,  29,  21,  44,  40,  31,  44,   7,
                     -17,  16,  -2,  15,  14,   0,  15, -13,
                     -26,   3,  10,   9,   6,   1,   0, -23,
                     -22,   9,   5, -11, -10,  -2,   3, -19,
                     -31,   8,  -7, -37, -36, -14,   3, -31,
                       0,   0,   0,   0,   0,   0,   0,   0
                ],
            'n': [ 
                    -66, -53, -75, -75, -10, -55, -58, -70,
                     -3,  -6, 100, -36,   4,  62,  -4, -14,
                     10,  67,   1,  74,  73,  27,  62,  -2,
                     24,  24,  45,  37,  33,  41,  25,  17,
                     -1,   5,  31,  21,  22,  35,   2,   0,
                    -18,  10,  13,  22,  18,  15,  11, -14,
                    -23, -15,   2,   0,   2,   0, -23, -20,
                    -74, -23, -26, -24, -19, -35, -22, -69
                ],
            'b': [ 
                    -59, -78, -82, -76, -23,-107, -37, -50,
                    -11,  20,  35, -42, -39,  31,   2, -22,
                     -9,  39, -32,  41,  52, -10,  28, -14,
                     25,  17,  20,  34,  26,  25,  15,  10,
                     13,  10,  17,  23,  17,  16,   0,   7,
                     14,  25,  24,  15,   8,  25,  20,  15,
                     19,  20,  11,   6,   7,   6,  20,  16,
                     -7,   2, -15, -12, -14, -15, -10, -10
                ],
            'r': [  
                     35,  29,  33,   4,  37,  33,  56,  50,
                     55,  29,  56,  67,  55,  62,  34,  60,
                     19,  35,  28,  33,  45,  27,  25,  15,
                      0,   5,  16,  13,  18,  -4,  -9,  -6,
                    -28, -35, -16, -21, -13, -29, -46, -30,
                    -42, -28, -42, -25, -25, -35, -26, -46,
                    -53, -38, -31, -26, -29, -43, -44, -53,
                    -30, -24, -18,   5,  -2, -18, -31, -32
                ],
            'q': [   
                      6,   1,  -8,-104,  69,  24,  88,  26,
                     14,  32,  60, -10,  20,  76,  57,  24,
                     -2,  43,  32,  60,  72,  63,  43,   2,
                      1, -16,  22,  17,  25,  20, -13,  -6,
                    -14, -15,  -2,  -5,  -1, -10, -20, -22,
                    -30,  -6, -13, -11, -16, -11, -16, -27,
                    -36, -18,   0, -19, -15, -15, -21, -38,
                    -39, -30, -31, -13, -31, -36, -34, -42
                ],
            'k': [  
                      4,  54,  47, -99, -99,  60,  83, -62,
                    -32,  10,  55,  56,  56,  55,  10,   3,
                    -62,  12, -57,  44, -67,  28,  37, -31,
                    -55,  50,  11,  -4, -19,  13,   0, -49,
                    -55, -43, -52, -28, -51, -47,  -8, -50,
                    -47, -42, -43, -79, -64, -32, -29, -32,
                     -4,   3, -14, -50, -57, -18,  13,   4,
                     17,  30,  -3, -14,   6,  -1,  40,  18
                ]
        }
        self.pst['P'] = self.reverse_chunks(self.pst['p'][::-1], 8)
        self.pst['N'] = self.reverse_chunks(self.pst['n'][::-1], 8)
        self.pst['B'] = self.reverse_chunks(self.pst['b'][::-1], 8)
        self.pst['R'] = self.reverse_chunks(self.pst['r'][::-1], 8)
        self.pst['Q'] = self.reverse_chunks(self.pst['q'][::-1], 8)
        self.pst['K'] = self.reverse_chunks(self.pst['k'][::-1], 8)

    def reset(self):
        """
        Resets the chessboard to the initial state.
        """
        self.board.reset()
        return self.get_board_state()
        
    def reverse_chunks(self, lst, chunk_size):
        rev_chunks = [lst[i:i + chunk_size][::-1] if i + chunk_size <= len(lst) else lst[i:][::-1] 
                for i in range(0, len(lst), chunk_size)]
        return [item for sublist in rev_chunks for item in sublist]
    
    def read_all_moves(self):
        f = open("all_moves.txt", "r")
        all_moves =  f.read().splitlines()
        f.close()
        
        all_moves_dict = {}
        all_moves_list = []
        for i in range(len(all_moves)):
            all_moves_dict[all_moves[i]] = i
            move_tokens = all_moves[i].split(" ")
            all_moves_list.append([int(move_tokens[0]), int(move_tokens[1])])
        
        return all_moves_dict, all_moves_list

    def get_board_state(self):
        """
        Converts the board state to a numerical format that can be used as input to the neural network.
        """
        # Example: convert the board to a numeric representation
        board_state = np.zeros(64)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                board_state[i] = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                                              'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}[piece.symbol()]
        return board_state
        
    def get_legal_moves_encoded(self):
        """
        Returns a list of all legal moves encoded as unique numbers.
        """
        legal_moves = list(self.board.legal_moves)
        encoded_moves = [self.encode_move(move) for move in legal_moves]
        return encoded_moves
        
    def encode_move(self, move):
        move_str = str(move.from_square) + " " + str(move.to_square)
        return self.all_moves[move_str]
    
    def invert(self, state):
        state = -state
        state = state[::-1]
        
        return state
    
    def step(self, move_index):
        """
        Makes a move on the board.
        :param action: a move in UCI format (e.g., 'e2e4')
        :return: new state, and game over status
        """
        from_square = self.all_moves_list[move_index][0]
        to_square = self.all_moves_list[move_index][1]
        move = self.board.find_move(from_square, to_square)
        board_score_before = self.calculate_score()
        self.board.push(move)
        game_over = self.board.is_game_over()
        new_state = self.invert(self.get_board_state())
        
        if not game_over and len(self.board.move_stack) > 200:
            return new_state, True
        return new_state, game_over
    
    def get_legal_moves(self):
        return self.board.legal_moves
    
    def get_game_result(self):
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                return 1
            elif outcome.winner == chess.BLACK:
                return 0
        # draw
        return 0.5
    
    def calculate_score(self):
        """
        Calculate the score of a position
        """
        
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                return 60000
            elif outcome.winner == chess.BLACK:
                return -60000
            else:
                # Draw
                return 0
        else:
            score = 0
            white_score = 0
            black_score = 0
            for i in range(64):
                piece = self.board.piece_at(i)
                if piece:
                    piece_score = self.piece_weights[piece.symbol()]
                    score += self.piece_weights[piece.symbol()]
                    if (piece_score > 0):
                        white_score += piece_score
                    elif (piece_score < 0):
                        black_score -= piece_score
            position_score = 0
            for i in range(64):
                piece = self.board.piece_at(i)
                if piece:
                    # Calculate position score
                    piece_symbol = piece.symbol()                    
                    temp_position_score = self.pst[piece_symbol][i]
                    if (piece.color == chess.WHITE):
                        position_score += temp_position_score
                    else:
                        position_score -= temp_position_score
            
            return score + position_score

    def render(self):
        """
        Prints the current board state. Useful for debugging.
        """
        print(self.board)
        