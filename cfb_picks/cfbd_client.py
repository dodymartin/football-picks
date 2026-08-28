import os

import requests


class CFBDClient:
    BASE_URL = "https://api.collegefootballdata.com"

    def __init__(self, api_key=None, session=None):
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "CFBD_API_KEY not set. Get a free key at "
                "https://collegefootballdata.com/key and set it in your "
                "environment or a .env file."
            )
        self.session = session or requests.Session()

    def _get(self, path, params=None):
        response = self.session.get(
            f"{self.BASE_URL}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_fbs_teams(self, year):
        return self._get("/teams/fbs", {"year": year})

    def get_games(self, year, season_type="regular"):
        return self._get("/games", {"year": year, "seasonType": season_type})

    def get_lines(self, year, season_type="regular"):
        return self._get("/lines", {"year": year, "seasonType": season_type})

    def get_advanced_game_stats(self, year, season_type="regular"):
        return self._get(
            "/stats/game/advanced", {"year": year, "seasonType": season_type}
        )
