import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
# better table views
from IPython.display import display
# Chess game parser
import chess.pgn
from io import StringIO


# Utility functions

def fetch_json(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return {}

class PlayerGamesOverview:
    def __init__(self, username):
        self.link = f'https://api.chess.com/pub/player/{username}/games'
        self.username = username
        self.game_values = {
                            'checkmated': -1, 
                            'timeout': -1, 
                            'resigned': -1, 
                            'abandoned': -1, 
                            'draw': 0, 
                            'agreed': 0, 
                            'repetition': 0, 
                            'timevsinsufficient': 0, 
                            'insufficient': 0, 
                            'stalemate': 0, 
                            'win': 1
                            }
    
    def get_data(self, year, month):
        month = str(month).zfill(2)  # Ensure month is two digits
        url = f'{self.link}/{year}/{month}'
        
        self.create_player_games_table(fetch_json(url))
        # response = requests.get(url, headers=self.headers)
        # if response.status_code == 200:    
        #     data = response.json()
        #     return self.create_player_games_table(data)
        # else:
        #  print(f"Error: {response.status_code}")
    
    def create_player_games_table(self, df):
        games_data = []
        for game in df['games']:
            white_player = game["white"]["username"].lower()
            is_white = white_player == self.username.lower()
            opponent_data = game["black"] if is_white else game["white"]

            game_info = {
                "Date": game.get("end_time"),
                "Result": game["white"]["result"] if is_white else game["black"]["result"],
                "Color": "White" if is_white else "Black",
                "Opponent": opponent_data["username"],
                "Opponent Rating": opponent_data["rating"],
                "Opening": game.get("eco", "").split("/")[-1] if game.get("eco") else None,
                "Time Class": game.get("time_class", "")
            }
            games_data.append(game_info)

        # Create a DataFrame
        self.df_games = pd.DataFrame(games_data)
        self.df_games["Date"] = pd.to_datetime(self.df_games["Date"], unit='s')
        self.df_games['game_value'] = self.df_games['Result'].map(self.game_values).astype(int)
        
        return self.df_games
    
    def analyze_res_vs_opening(self):
        
        result_vs_opening = self.df_games.pivot_table(
                                        index='Opening',
                                        columns='game_value',
                                        values='Opponent',
                                        aggfunc='count',
                                        margins=True,
                                        margins_name='Sum').fillna(0).astype(int).reset_index()
        
        # top weak and strong games
        top_weak = result_vs_opening.sort_values(by=-1, ascending=False)[0:11]
        top_strong = result_vs_opening.sort_values(by=1, ascending=False)[0:11]
        
        # win rate
        sum_row = result_vs_opening[result_vs_opening['Opening'] == 'Sum']
        total_wins = sum_row[1].values[0] if 1 in sum_row else 0
        total_games = sum_row['Sum'].values[0] if 'Sum' in sum_row else 1
        win_rate = total_wins / total_games
        
        return {
        "summary_table": result_vs_opening,
        "top_weak_openings": top_weak,
        "top_strong_openings": top_strong,
        "win_rate": win_rate,
    }
        
    def analyze_res_vs_color(self):
        self.color_vs_result = self.df_games.pivot_table(
                                index='Color', 
                                columns='game_value', 
                                values='Opponent', 
                                aggfunc='count', 
                                margins=True, 
                                margins_name='Sum').fillna(0).astype(int)
        
        # Per col 
        color_vs_result_total_col = round(self.color_vs_result/self.color_vs_result.loc['Sum'], 4) * 100
        # Per row 
        color_vs_result_total_row = self.color_vs_result.div(self.color_vs_result['Sum'], axis=0)
        color_vs_result_total = round(self.color_vs_result/self.color_vs_result.loc['Sum','Sum'], 4) * 100
        
        return {
            "summary_table": self.color_vs_result,
            "share_of_total_col": color_vs_result_total_col,
            "share_of_total_row": color_vs_result_total_row,
            "share_of_total": color_vs_result_total,
            "stat_analysis": self.test_color_hypotheses(),
            "win_rate": {
                "white": self.color_vs_result.loc['White', 1] / self.color_vs_result.loc['White', 'Sum'],
                "black": self.color_vs_result.loc['Black', 1] / self.color_vs_result.loc['Black', 'Sum'],
            }
        }

    
    def test_color_hypotheses(self):
        win_counts = [self.color_vs_result.loc['White', 1], self.color_vs_result.loc['Black', 1]]
        n_obs = [self.color_vs_result.loc['White', 'Sum'], self.color_vs_result.loc['Black', 'Sum']]  
        loss_counts = [self.color_vs_result.loc['White', -1], self.color_vs_result.loc['Black', -1]]
        white_not_win = self.color_vs_result.loc['White', -1] + self.color_vs_result.loc['White', 0]
        black_not_win = self.color_vs_result.loc['Black', -1] + self.color_vs_result.loc['Black', 0]
        table = [[self.color_vs_result.loc['White', 1], white_not_win],
                [self.color_vs_result.loc['Black', 1], black_not_win]]
        return{
    # Hypothesis review
    # Hypothesis 1: Chi-Square Test of Independence
    # Do game outcomes depend on piece color (Black vs White)? / Color of piece will impact your game result!
    "chi2_p": chi2_contingency(self.color_vs_result.loc[['Black', 'White'], [-1, 0, 1]]),
    # Hypothesis 2: Two-Proportion Z-Test for Win Rates
    # Do White and Black have significantly different win rates? / White and Black win rates are significantly different!
    "ztest_win_p": proportions_ztest(count=win_counts, nobs=n_obs),
    # 3. One-tailed Z-test for loss rates 
    # Is White more likely to lose than Black? / White is more likely to lose than Black
    "ztest_loss_p":  proportions_ztest(count=loss_counts, nobs=n_obs, alternative='larger'),
    # Hypothesis 4: Fisher’s Exact Test (Win vs Not-Win)
    # Are odds of winning as White greater than Black? / Odds of winning as White are greater than Black
    "fisher_p": fisher_exact(table, alternative='greater')
    }

    
    def opponent_rating_dist(self):
        temp_rating = self.df_games['Opponent Rating']
        ax = sns.histplot(temp_rating, bins=20)
        ax.set_title('Opponent Rating Distribution')
        plt.xlabel('Opponent Rating')
        plt.ylabel('Frequency')
        plt.show()
    # TODO -- when we want the user to choose now we go an easy way to build the structure.
# self.months = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April',
#     5: 'May', 6: 'June', 7: 'July', 8: 'August',
#     9: 'September', 10: 'October', 11: 'November', 12: 'December'
# }

class GameArchives:
    """
    A class to play around with Chess.com player game archives.
    """
    
    def __init__(self, username):
        self.link = f'https://api.chess.com/pub/player/{username}/games/archives'
        self.username = username
        self.archives = []
        
    
    def get_latest_games(self):
        """
        Fetch latest month's game archives of the player.
        """
        game_list = fetch_json(self.link)
        if not game_list:
            raise ValueError("Could not fetch archives.")
        
        last_url = game_list['archives'][-1]
        game_data = fetch_json(last_url)
        if not game_data:
            raise ValueError("Could not fetch latest games.")
        
        self.game = game_data
        return self.game
    
    def add_moves_to_game(self):
        """
        Add moves to the game data.
        """
        games = self.game['games']
        
        for game in games:
            game['Moves'] = self.game_parser(game['pgn'])['Moves']
            
        return self.game
        

    @staticmethod
    def game_parser(pgn_text):
        """
        Parse a PGN string and return the game object.
        Requires a valid PGN string.
        """
        pgn_io = StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        
        Moves = GameArchives.iterate_moves(game)
        
        return {
            'game': game,
            'Event': game.headers.get("Event"),
            'White': game.headers.get("White"),
            'Black': game.headers.get("Black"),
            'Result': game.headers.get("Result"),
            'Date': game.headers.get("Date"),
            'White Elo': game.headers.get("WhiteElo"),
            'Black Elo': game.headers.get("BlackElo"),
            'Link': game.headers.get("Link"),
            'Moves': Moves
        }


    @staticmethod
    def iterate_moves(game):
        """
        Iterate through moves in a game.
        Each move_id is shared by a White and Black move.
        """
        res = []
        board = game.board()
        full_move_id = 1  # incremented every 2 half-moves

        for i, move in enumerate(game.mainline_moves()):
            color = 'White' if board.turn == chess.WHITE else 'Black'

            res.append({
                'move_id': full_move_id,
                'color': color,
                'notation': board.san(move)
            })

            board.push(move)

            # Only increment after Black's move
            if color == 'Black':
                full_move_id += 1

        return res

    
    

if __name__ == "__main__":
    pgo = PlayerGamesOverview("Frezaeei")
    df = pgo.get_data(2025, 5)
    print(df.head())
    
    ga = GameArchives('Frezaeei')
    ga.get_latest_games()
    res = ga.add_moves_to_game()
    print(res)
