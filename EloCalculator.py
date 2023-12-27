import os
from ChessAgent import ChessAgent, load_model
from ChessEnvironment import ChessEnvironment

class EloCalculator:
    def __init__(self, k_factor=20):
        self.k = k_factor

    def expected_score(self, rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))

    def new_rating(self, rating, opponent_rating, score):
        expected = self.expected_score(rating, opponent_rating)
        return rating + self.k * (score - expected)

class ChessTournament:
    def __init__(self, games_per_match):
        self.bots = self.get_bots()
        self.games_per_match = games_per_match
        self.elo_calculator = EloCalculator()
        self.env = ChessEnvironment()

    def get_bots(self):
        bots = []
        files = os.listdir('.')
        filtered_files = [file for file in files if file.startswith("chess_model_games_")]
        for filename in filtered_files:
            agent = ChessAgent()
            load_model(agent, filename)
            bots.append(agent)
        
        return bots

    def play_game(self, bot1, bot2):
        # Implement the logic to play a game between bot1 and bot2
        # Return the result (1 if bot1 wins, 0 if bot2 wins, 0.5 for a draw)
        board_state = self.env.reset()
        done = False
        while not done:
            legal_moves = self.env.get_legal_moves_encoded()
            bot1_move = bot1.select_move_deterministic(board_state, legal_moves)
            board_state, done = self.env.step(bot1_move)
            if not done:
                legal_moves = self.env.get_legal_moves_encoded()
                bot2_move = bot2.select_move_deterministic(board_state, legal_moves)
                board_state, done = self.env.step(bot2_move)
        
        return self.env.get_game_result()
        

    def update_ratings(self, bot1, bot2, result):
        bot1_new_rating = self.elo_calculator.new_rating(bot1.rating, bot2.rating, result)
        bot2_new_rating = self.elo_calculator.new_rating(bot2.rating, bot1.rating, 1 - result)
        bot1.rating, bot2.rating = bot1_new_rating, bot2_new_rating

    def run_tournament(self):
        for i in range(len(self.bots)):
            for j in range(i + 1, len(self.bots)):
                for _ in range(int(self.games_per_match / 2)):
                    result = self.play_game(self.bots[i], self.bots[j])
                    self.update_ratings(self.bots[i], self.bots[j], result)
                for _ in range(int(self.games_per_match / 2)):
                    result = self.play_game(self.bots[j], self.bots[i])
                    self.update_ratings(self.bots[j], self.bots[i], result)

# Example Usage
tournament = ChessTournament(games_per_match=10)
tournament.run_tournament()

# Print the final ratings
for bot in tournament.bots:
    print(f"{bot.name}: {bot.rating}")