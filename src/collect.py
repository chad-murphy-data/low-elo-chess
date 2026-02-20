"""
Data collection from the Lichess API.

Strategy: snowball sampling.
1. Seed with a few users from player autocomplete
2. Download their blitz games (with evals + clocks)
3. Discover opponents at various ELO bands
4. Repeat until we have enough users per band

Rate limit: 1 request/sec with polite User-Agent header.
"""

import csv
import io
import json
import os
import re
import time
from pathlib import Path

import chess
import chess.pgn
import requests

BASE_URL = "https://lichess.org"
USER_AGENT = "chess-blunder-research (github.com/low-elo-chess)"
HEADERS = {"User-Agent": USER_AGENT}

ELO_BANDS = [
    (500, 700),
    (700, 900),
    (900, 1100),
    (1100, 1300),
    (1300, 1500),
]

# Targets
TARGET_USERS_PER_BAND = 200
TARGET_GAMES_PER_USER = 20


def _rate_limit():
    """Sleep to stay within Lichess rate limits."""
    time.sleep(1.1)


def _api_get(path, params=None, accept=None, stream=False):
    """Make a GET request to the Lichess API with rate limiting and retries."""
    url = f"{BASE_URL}{path}"
    headers = dict(HEADERS)
    if accept:
        headers["Accept"] = accept

    for attempt in range(4):
        try:
            resp = requests.get(
                url, params=params, headers=headers, stream=stream, timeout=30
            )
            if resp.status_code == 429:
                wait = 60  # Lichess asks for 60s on 429
                print(f"  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            _rate_limit()
            return resp
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"  Request error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  Failed after 4 attempts: {path}")
    return None


def get_user_rating(username):
    """Get a user's blitz rating. Returns (rating, num_games) or None."""
    resp = _api_get(f"/api/user/{username}", accept="application/json")
    if resp is None:
        return None
    try:
        data = resp.json()
        blitz = data.get("perfs", {}).get("blitz", {})
        rating = blitz.get("rating")
        games = blitz.get("games", 0)
        if rating is None:
            return None
        return (rating, games)
    except (json.JSONDecodeError, KeyError):
        return None


def find_elo_band(elo):
    """Return the ELO band tuple for a given rating, or None if out of range."""
    for low, high in ELO_BANDS:
        if low <= elo < high:
            return (low, high)
    return None


def seed_users_from_autocomplete(search_terms=None):
    """Find seed users via Lichess player autocomplete."""
    if search_terms is None:
        # Use a variety of short terms to get diverse users
        search_terms = [
            "ch", "ki", "pa", "ma", "jo", "da", "an", "mi",
            "sa", "al", "ra", "to", "be", "ca", "de", "el",
            "fi", "ga", "ha", "in", "ja", "ka", "la", "na",
        ]

    users = {}  # username -> rating
    print("Finding seed users via autocomplete...")
    for term in search_terms:
        resp = _api_get(
            "/api/player/autocomplete",
            params={"term": term, "object": "true", "nb": 15},
            accept="application/json",
        )
        if resp is None:
            continue
        try:
            results = resp.json()
            for user in results:
                username = user.get("id", user.get("name", ""))
                if not username:
                    continue
                # We need to look up their actual blitz rating
                info = get_user_rating(username)
                if info is None:
                    continue
                rating, num_games = info
                if num_games < 50:
                    continue  # skip inactive players
                band = find_elo_band(rating)
                if band is not None:
                    users[username] = rating
                    print(f"  Seed: {username} (blitz {rating})")
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return users


def discover_users_from_games(pgn_text, known_users):
    """Extract opponent usernames and ratings from PGN game headers."""
    discovered = {}
    pgn_io = io.StringIO(pgn_text)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        headers = game.headers
        for color in ["White", "Black"]:
            username = headers.get(color, "")
            elo_str = headers.get(f"{color}Elo", "")
            if not username or not elo_str:
                continue
            try:
                elo = int(elo_str)
            except ValueError:
                continue
            username_lower = username.lower()
            if username_lower in known_users:
                continue
            band = find_elo_band(elo)
            if band is not None:
                discovered[username_lower] = elo

    return discovered


def download_user_games(username, max_games=TARGET_GAMES_PER_USER):
    """Download recent rated blitz games for a user as PGN text."""
    resp = _api_get(
        f"/api/games/user/{username}",
        params={
            "perfType": "blitz",
            "rated": "true",
            "max": max_games,
            "evals": "true",
            "clocks": "true",
            "moves": "true",
            "tags": "true",
            "opening": "true",
            "sort": "dateDesc",
        },
        accept="application/x-chess-pgn",
    )
    if resp is None:
        return None
    return resp.text


def save_checkpoint(checkpoint_path, users_by_band, processed_users, games_dir):
    """Save collection progress to a checkpoint file."""
    data = {
        "users_by_band": {
            f"{lo}-{hi}": list(users)
            for (lo, hi), users in users_by_band.items()
        },
        "processed_users": list(processed_users),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(checkpoint_path):
    """Load collection progress from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path) as f:
        data = json.load(f)
    users_by_band = {}
    for key, users in data["users_by_band"].items():
        lo, hi = key.split("-")
        users_by_band[(int(lo), int(hi))] = set(users)
    processed_users = set(data["processed_users"])
    return users_by_band, processed_users


def collect_data(data_dir="data", target_users=TARGET_USERS_PER_BAND,
                 target_games=TARGET_GAMES_PER_USER, max_iterations=50):
    """
    Main collection loop. Snowball samples users by ELO band and downloads
    their games.

    Args:
        data_dir: Directory to store PGN files and checkpoints
        target_users: Target number of users per ELO band
        target_games: Number of games to download per user
        max_iterations: Maximum snowball iterations to prevent infinite loops
    """
    data_path = Path(data_dir)
    games_dir = data_path / "pgn"
    games_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = data_path / "collection_checkpoint.json"

    # Try to resume from checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        users_by_band, processed_users = checkpoint
        print(f"Resumed from checkpoint: {len(processed_users)} users processed")
    else:
        users_by_band = {band: set() for band in ELO_BANDS}
        processed_users = set()

    def band_status():
        return {
            f"{lo}-{hi}": len(users) for (lo, hi), users in users_by_band.items()
        }

    # Phase 1: Seed users
    if sum(len(u) for u in users_by_band.values()) == 0:
        seed_users = seed_users_from_autocomplete()
        for username, rating in seed_users.items():
            band = find_elo_band(rating)
            if band:
                users_by_band[band].add(username)
        print(f"After seeding: {band_status()}")

    # Phase 2: Snowball sampling
    iteration = 0
    while iteration < max_iterations:
        # Check if we have enough users in all bands
        all_full = all(
            len(users) >= target_users for users in users_by_band.values()
        )
        if all_full:
            print("All ELO bands have enough users!")
            break

        # Find the band with fewest users that still needs more
        neediest_band = min(
            [b for b in ELO_BANDS if len(users_by_band[b]) < target_users],
            key=lambda b: len(users_by_band[b]),
            default=None,
        )
        if neediest_band is None:
            break

        # Pick an unprocessed user from the neediest band
        unprocessed = users_by_band[neediest_band] - processed_users
        if not unprocessed:
            # Try to find users from other bands that might discover
            # opponents in the neediest band
            for band in ELO_BANDS:
                unprocessed = users_by_band[band] - processed_users
                if unprocessed:
                    break
            if not unprocessed:
                print("No more unprocessed users. Need more seeds.")
                # Try to get more seeds
                extra_seeds = seed_users_from_autocomplete(
                    [f"{chr(a)}{chr(b)}" for a in range(ord('a'), ord('z'))
                     for b in range(ord('a'), ord('d'))][:20]
                )
                for username, rating in extra_seeds.items():
                    band = find_elo_band(rating)
                    if band:
                        users_by_band[band].add(username)
                if sum(len(u) - len(processed_users & u)
                       for u in users_by_band.values()) == 0:
                    print("Cannot find more users. Stopping.")
                    break
                continue

        username = next(iter(unprocessed))
        processed_users.add(username)

        print(f"\n[Iter {iteration}] Processing {username} "
              f"(band {neediest_band[0]}-{neediest_band[1]})")
        print(f"  Band status: {band_status()}")

        # Download games
        pgn_file = games_dir / f"{username}.pgn"
        if pgn_file.exists():
            with open(pgn_file) as f:
                pgn_text = f.read()
        else:
            pgn_text = download_user_games(username, max_games=target_games)
            if pgn_text is None or len(pgn_text.strip()) == 0:
                print(f"  No games found for {username}")
                iteration += 1
                continue
            with open(pgn_file, "w") as f:
                f.write(pgn_text)

        # Discover opponents
        all_known = set()
        for users in users_by_band.values():
            all_known.update(users)
        new_users = discover_users_from_games(pgn_text, all_known)

        added = 0
        for new_username, rating in new_users.items():
            band = find_elo_band(rating)
            if band and len(users_by_band[band]) < target_users:
                users_by_band[band].add(new_username)
                added += 1

        print(f"  Discovered {len(new_users)} new users, added {added} to bands")

        # Checkpoint every 10 iterations
        if iteration % 10 == 0:
            save_checkpoint(checkpoint_path, users_by_band, processed_users, games_dir)

        iteration += 1

    # Final checkpoint
    save_checkpoint(checkpoint_path, users_by_band, processed_users, games_dir)

    # Phase 3: Download games for all users we haven't processed yet
    print("\n--- Phase 3: Downloading remaining games ---")
    all_users_to_download = set()
    for users in users_by_band.values():
        all_users_to_download.update(users)

    for i, username in enumerate(all_users_to_download):
        pgn_file = games_dir / f"{username}.pgn"
        if pgn_file.exists():
            continue
        print(f"  [{i+1}/{len(all_users_to_download)}] Downloading {username}...")
        pgn_text = download_user_games(username, max_games=target_games)
        if pgn_text and len(pgn_text.strip()) > 0:
            with open(pgn_file, "w") as f:
                f.write(pgn_text)

        if i % 20 == 0:
            save_checkpoint(
                checkpoint_path, users_by_band, processed_users, games_dir
            )

    save_checkpoint(checkpoint_path, users_by_band, processed_users, games_dir)

    # Summary
    print("\n=== Collection Summary ===")
    total_games = 0
    for band, users in sorted(users_by_band.items()):
        n_users = len(users)
        n_games = sum(
            1
            for u in users
            if (games_dir / f"{u}.pgn").exists()
        )
        print(f"  {band[0]}-{band[1]}: {n_users} users, {n_games} with games")
        total_games += n_games
    print(f"  Total PGN files: {total_games}")

    return users_by_band


if __name__ == "__main__":
    collect_data()
