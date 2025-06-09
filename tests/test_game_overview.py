import json
import re
from pathlib import Path

import pandas as pd
import pytest

import game_overview
from game_overview import PlayerGamesOverview


def _load_local_games():
    text = Path('2025-05_chess_games.json').read_text()
    pattern = re.compile(r'"pgn": "((?:\\.|[^"\\])*?)"', re.DOTALL)
    fixed = pattern.sub(lambda m: '"pgn": "' + m.group(1).replace("\n", "\\n") + '"', text)
    return json.loads(fixed)


def test_get_data_and_win_rate(monkeypatch):
    data = _load_local_games()

    monkeypatch.setattr(game_overview, 'fetch_json', lambda url: data)

    pgo = PlayerGamesOverview('Frezaeei')
    df = pgo.get_data(2025, 5)
    assert isinstance(df, pd.DataFrame)

    stats = pgo.win_rate()
    assert isinstance(stats, dict)
    assert stats['total_games'] == len(df)
    total = stats['total_wins'] + stats['total_losses'] + stats['total_draws']
    assert total == len(df)

