from chess_agent import ChessAgent
from chess_environment import ChessEnvironment


class GenerateTrainingDataInput():
    def __init__(self, agent: ChessAgent, env: ChessEnvironment, opponent: ChessAgent, is_white, alpha):
        self.agent = agent
        self.env = env
        self.opponent = opponent
        self.is_white = is_white
        self.alpha = alpha