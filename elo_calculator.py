from multiprocessing import Pool
import os
import model_utilities
import constants
import multithreading
from chess_agent import ChessAgent
from chess_environment import ChessEnvironment

class EloCalculator:
    def __init__(self, k_factor=20):
        self.k = k_factor
        self.elos = model_utilities.load_values('elos.txt')
        self.elos_v2 = model_utilities.load_values('elos_v2.txt')

    def expected_score(self, rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))

    def new_rating(self, rating, opponent_rating, score):
        expected = self.expected_score(rating, opponent_rating)
        return rating + self.k * (score - expected)
    
    def _baseline_to_zero(self):
        baseline = self.elos['0']
        for key in self.elos:
            self.elos[key] -= baseline

        baseline_v2 = self.elos_v2['0']
        for key in self.elos_v2:
            self.elos_v2[key] -= baseline_v2

    def calculate_elo(self, model: ChessAgent, models):
        opponents = self._get_opponents(model.name, models)
        for opponent in opponents:
            game_results = self._play_games_multithreaded(model, opponent)

            for game_result, position_result in game_results:
                self._update_ratings(model, opponent, game_result)
                self._update_ratings_v2(model, opponent, position_result)
        
        self._baseline_to_zero()
        model_utilities.save_values(self.elos, 'elos.txt')
        model_utilities.save_values(self.elos_v2, 'elos_v2.txt')

    def _update_ratings(self, bot1, bot2, result):
        bot1_new_rating = self.new_rating(self.elos.get(bot1.name, 0), self.elos.get(bot2.name, 0), result)
        bot2_new_rating = self.new_rating(self.elos.get(bot2.name, 0), self.elos.get(bot1.name, 0), 1 - result)
        self.elos[bot1.name] = bot1_new_rating
        self.elos[bot2.name] = bot2_new_rating

    def _update_ratings_v2(self, bot1, bot2, result):
        bot1_new_rating = self.new_rating(self.elos_v2.get(bot1.name, 0), self.elos_v2.get(bot2.name, 0), result)
        bot2_new_rating = self.new_rating(self.elos_v2.get(bot2.name, 0), self.elos_v2.get(bot1.name, 0), 1 - result)
        self.elos_v2[bot1.name] = bot1_new_rating
        self.elos_v2[bot2.name] = bot2_new_rating
    
    def _play_games_multithreaded(self, model, opponent):
        play_game_inputs = [[model_utilities.copy_model(model), model_utilities.copy_model(opponent), ChessEnvironment(), bool(x % 2)] for x in range(2)]
        game_results = multithreading.multithreading_pool(model_utilities.play_game, play_game_inputs)
        
        return game_results

    def _get_opponents(self, agent_name, models):
        opponents = []
        files = os.listdir('models/.')
        filtered_files = sorted([int(file) for file in files if str(file) != agent_name])
        filtered_files = [str(file) for file in filtered_files if str(file) != agent_name]
        for filename in filtered_files:
            opponent = model_utilities.load_model(models, filename)
            opponents.append(opponent)
        
        return opponents
    
    def _pick_evenly_spaced(self, lst):
        """
        Selects 10 numbers from a list with even spacing.
        If the list has fewer than 10 elements, it returns the entire list.
        """
        n = len(lst)
        
        # If the list has fewer than 10 elements, return the entire list
        if n <= 10:
            return lst
        
        # Calculate the interval for picking elements
        interval = n // 10
        
        # Select elements at evenly spaced intervals
        return [lst[i * interval] for i in range(10)]
    
    def recalculate_elos_from_scratch(self):
        files = os.listdir('models/.')
        models = {}
        for file in files:
            model_utilities.load_model(models, file)

        self.elos = {}
        self.elos_v2 = {}

        # Get all bot keys
        bot_keys = sorted([int(key) for key in models.keys()])

        # Iterate over each bot
        for i in range(len(bot_keys)):
            bot1_key = str(bot_keys[i])
            print(bot1_key)
            bot1 = models[bot1_key]

            # Iterate over each other bot, starting from the next in the list
            for j in range(i + 1, len(bot_keys)):
                bot2_key = str(bot_keys[j])
                bot2 = models[bot2_key]

                game_results = self._play_games_multithreaded(bot1, bot2)

                for game_result, position_result in game_results:
                    self._update_ratings(bot1, bot2, game_result)
                    self._update_ratings_v2(bot1, bot2, position_result)

        self._baseline_to_zero()
        model_utilities.save_values(self.elos, 'elos.txt')
        model_utilities.save_values(self.elos_v2, 'elos_v2.txt')

if __name__ == '__main__':
    calculator = EloCalculator()
    calculator.recalculate_elos_from_scratch()
    