"""
Stockfish local evaluation — fill in missing evals for all PGN games.

Replays each game move-by-move, evaluates each position with Stockfish,
and writes eval results to a JSON checkpoint file. The feature rebuild
step then merges these evals into the CSV.

Usage:
    python -m src.stockfish_eval              # Run with defaults (depth 10)
    python -m src.stockfish_eval --depth 8    # Faster, slightly less accurate
    python -m src.stockfish_eval --resume     # Resume from checkpoint

Eval convention: centipawns from White's perspective (positive = White better).
Mate scores are mapped to ±10_000.
"""

import json
import os
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

SF_PATH = r"C:\Users\chadm\Desktop\low elo chess\stockfish_dir\stockfish\stockfish-windows-x86-64-avx2.exe"

CHECKPOINT_FILE = "data/stockfish_evals.json"
CHECKPOINT_EVERY = 25  # Save progress every N games


def score_to_cp(score):
    """Convert a chess.engine Score to centipawns from White's perspective."""
    if score.is_mate():
        return 10000 if score.white().mate() > 0 else -10000
    cp = score.white().score()
    return cp if cp is not None else None


def load_checkpoint(path):
    """Load existing eval checkpoint. Returns dict of game_id -> list of evals."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(path, data):
    """Save eval checkpoint atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def eval_game(engine, game, depth=10):
    """Evaluate every position in a game.

    Returns a list of centipawn evals (from White's perspective),
    one per ply. The i-th eval is the position AFTER the i-th move.
    We also return the starting position eval as index 0.
    """
    board = game.board()
    moves = list(game.mainline_moves())

    evals = []

    # Eval starting position (before move 1)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    evals.append(score_to_cp(info["score"]))

    for move in moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        evals.append(score_to_cp(info["score"]))

    return evals


def get_game_id(game):
    """Extract game ID from PGN headers."""
    site = game.headers.get("Site", "")
    gid = site.split("/")[-1] if site else ""
    if not gid:
        gid = game.headers.get("Event", "unknown")
    return gid


def run_stockfish_evals(data_dir="data", depth=10, resume=True):
    """Evaluate all games missing evals using local Stockfish."""
    data_path = Path(data_dir)
    pgn_dir = data_path / "pgn"
    checkpoint_path = str(data_path / "stockfish_evals.json")

    # Load checkpoint
    evals_db = load_checkpoint(checkpoint_path) if resume else {}
    print(f"Checkpoint: {len(evals_db)} games already evaluated")

    # Collect all games from PGN files
    pgn_files = sorted(pgn_dir.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files")

    # Count total games first
    total_games = 0
    games_to_eval = []

    for pgn_file in pgn_files:
        with open(pgn_file, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                total_games += 1
                gid = get_game_id(game)
                mainline = list(game.mainline_moves())
                if len(mainline) < 20:
                    continue  # Skip short games (same filter as features.py)
                if gid in evals_db:
                    continue  # Already evaluated
                games_to_eval.append((pgn_file.name, gid, game))

    print(f"Total games: {total_games}")
    print(f"Games needing eval: {len(games_to_eval)}")
    print(f"Games already done: {len(evals_db)}")
    print()

    if not games_to_eval:
        print("Nothing to do!")
        return evals_db

    # Start Stockfish
    threads = max(1, (os.cpu_count() or 4) - 2)
    print(f"Starting Stockfish (depth={depth}, threads={threads}, hash=256MB)")
    engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    engine.configure({"Threads": threads, "Hash": 256})

    start_time = time.time()
    games_done = 0
    positions_done = 0

    try:
        for i, (pgn_name, gid, game) in enumerate(games_to_eval):
            game_start = time.time()
            game_evals = eval_game(engine, game, depth=depth)
            game_time = time.time() - game_start
            n_positions = len(game_evals)

            evals_db[gid] = game_evals
            games_done += 1
            positions_done += n_positions

            # Progress
            elapsed = time.time() - start_time
            rate = positions_done / elapsed if elapsed > 0 else 0
            remaining = len(games_to_eval) - games_done
            eta_positions = remaining * (n_positions)  # rough estimate
            eta_seconds = eta_positions / rate if rate > 0 else 0

            if games_done % 5 == 0 or games_done <= 3:
                print(
                    f"  [{games_done}/{len(games_to_eval)}] {gid} "
                    f"({n_positions} pos in {game_time:.1f}s) "
                    f"| {rate:.0f} pos/s | ETA: {eta_seconds/60:.0f}m"
                )

            # Checkpoint
            if games_done % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_path, evals_db)
                print(f"  ** Checkpoint saved ({len(evals_db)} games) **")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
    finally:
        save_checkpoint(checkpoint_path, evals_db)
        engine.quit()

    total_time = time.time() - start_time
    print(f"\nDone! {games_done} games, {positions_done} positions in {total_time/60:.1f} minutes")
    print(f"Checkpoint saved: {checkpoint_path} ({len(evals_db)} games total)")

    return evals_db


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stockfish evals on PGN games")
    parser.add_argument("--depth", type=int, default=10, help="Search depth (default: 10)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    run_stockfish_evals(
        data_dir=args.data_dir,
        depth=args.depth,
        resume=not args.no_resume,
    )
