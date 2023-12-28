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
        self.elos = model_utilities.load_ratings()

    def expected_score(self, rating, opponent_rating):
        return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))

    def new_rating(self, rating, opponent_rating, score):
        expected = self.expected_score(rating, opponent_rating)
        return rating + self.k * (score - expected)
    
    def calculate_elo(self, model: ChessAgent, models):
        opponents = self._get_opponents(model.name, models)
        for opponent in opponents:
            game_results = self._play_games_multithreaded(model, opponent)

            for game_result in game_results:
                self._update_ratings(model, opponent, game_result)

        model_utilities.save_ratings(self.elos)
        

    def _update_ratings(self, bot1, bot2, result):
        bot1_new_rating = self.new_rating(self.elos.get(bot1.name, 0), self.elos.get(bot2.name, 0), result)
        bot2_new_rating = self.new_rating(self.elos.get(bot2.name, 0), self.elos.get(bot1.name, 0), 1 - result)
        self.elos[bot1.name] = bot1_new_rating
        self.elos[bot2.name] = bot2_new_rating
    
    def _play_games_multithreaded(self, model, opponent):
        play_game_inputs = [[model_utilities.copy_model(model), model_utilities.copy_model(opponent), ChessEnvironment(), bool(x % 2)] for x in range(constants.THREADS)]
        game_results = multithreading.multithreading_pool(model_utilities.play_game, play_game_inputs)
        
        return game_results

    def _get_opponents(self, agent_name, models):
        opponents = []
        files = os.listdir('models/.')
        filtered_files = sorted([int(file) for file in files if str(file) != agent_name])
        filtered_files = [str(file) for file in files if str(file) != agent_name]
        filtered_files = self._pick_evenly_spaced(filtered_files)
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
