from multiprocessing import Pool
import os
import model_utilities
import constants
from chess_agent import ChessAgent, load_model
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
        bot1_new_rating = self.new_rating(bot1.rating, bot2.rating, result)
        bot2_new_rating = self.new_rating(bot2.rating, bot1.rating, 1 - result)
        self.elos[bot1.name] = bot1_new_rating
        self.elos[bot2.name] = bot2_new_rating
    
    def _play_games_multithreaded(self, model, opponent):
        play_game_input = [[model, opponent, ChessEnvironment(), bool(x % 2)] for x in range(constants.THREADS * 2)]
        with Pool(processes=constants.THREADS) as p:
            game_results = p.map(self.play_game, play_game_input)
        
        return game_results

    def play_game(self, agent: ChessAgent, opponent: ChessAgent, env: ChessEnvironment, is_switch):
        env.reset()

        if is_switch:
            bot1 = opponent
            bot2 = agent
        else:
            bot1 = agent
            bot2 = opponent

        outcome = None
        while outcome is None:
            _, _, _, outcome  = model_utilities.make_move(env, bot1, True)
            if not outcome:
                _, _, _, outcome  = model_utilities.make_move(env, bot2, True)

        # (1 if bot1 wins, 0 if bot2 wins, 0.5 for a draw)
        game_result = env.get_game_result()
        if is_switch:
            game_result = 1 - game_result

        return game_result


    def _get_opponents(self, agent_name, models):
        opponents = []
        files = os.listdir('models/.')
        filtered_files = sorted([int(file) for file in files if str(file) != agent_name])
        filtered_files = [str(file) for file in files if str(file) != agent_name]
        filtered_files = self._pick_evenly_spaced(filtered_files)
        for filename in filtered_files:
            opponent = model_utilities.load_model(models, filename, self.elos)
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
