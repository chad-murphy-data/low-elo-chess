#!/usr/bin/env python3
"""
Stonefish Tournament -- Bot vs Bot round-robin with full logging.

Runs a round-robin tournament between the 500, 700, and 900 ELO bots.
Each pairing plays N games (alternating colors). Every move is logged
to the SQLite database (data/stonefish_games.db) for analysis.

Usage:
    python tournament.py                    # 10 games per pairing (default)
    python tournament.py --games 20         # 20 games per pairing
    python tournament.py --pairings 500,700 # only 500 vs 700

Output:
    - SQLite logs in data/stonefish_games.db (every move, eval, mechanism)
    - JSON summary in data/tournament_results.json
    - Console progress with live results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess

from src.engine import MaiaEngine, StockfishEngine
from src.bot import StonefishBot
from src.game_state import GameState
from src.game_logger import GameLogger


# ---------------------------------------------------------------------------
# Play a single game between two bots
# ---------------------------------------------------------------------------

def play_game(
    white_bot: StonefishBot,
    black_bot: StonefishBot,
    white_elo: int,
    black_elo: int,
    logger: GameLogger,
    max_moves: int = 200,
    verbose: bool = True,
) -> Dict:
    """Play a single game between two Stonefish bots.

    Both bots share the same Maia engine (thread-safe for sequential use)
    and Stockfish engine.

    Returns:
        Dict with game result data.
    """
    # Initialize game states
    gs_white = GameState(bot_color=chess.WHITE)
    gs_black = GameState(bot_color=chess.BLACK)

    # Start game log
    game_id = logger.start_game(
        bot_elo=white_elo,
        bot_color="white",
        opponent_type=f"stonefish_{black_elo}",
    )

    move_list = []
    mechanisms_white = []
    mechanisms_black = []
    blunders_white = 0
    blunders_black = 0

    game_start = time.time()

    for full_move in range(1, max_moves + 1):
        # === WHITE's turn ===
        if gs_white.board.is_game_over():
            break

        fen_before = gs_white.board.fen()
        eval_before = white_bot.evaluate_position(gs_white)

        move_w, meta_w = white_bot.select_move(gs_white)
        san_w = gs_white.board.san(move_w)
        record_w = gs_white.push_move(move_w, eval_before=eval_before)

        # Get eval after
        eval_after = white_bot.evaluate_position(gs_white)
        record_w.eval_after = eval_after

        # Compute cp_loss
        cp_loss = None
        if eval_before is not None and eval_after is not None:
            cp_loss = max(0, eval_before - eval_after)
        was_blunder = cp_loss is not None and cp_loss > 100
        if was_blunder:
            blunders_white += 1

        # Log white's move
        logger.log_move(
            game_id=game_id,
            move_number=gs_white.move_number,
            ply=gs_white.ply,
            player=f"stonefish_{white_elo}",
            fen_before=fen_before,
            move_uci=move_w.uci(),
            move_san=san_w,
            piece_moved=record_w.piece_name,
            is_capture=record_w.is_capture,
            is_check=record_w.is_check,
            eval_before=eval_before,
            eval_after=eval_after,
            cp_loss=cp_loss,
            was_blunder=was_blunder,
            mechanism=meta_w.get("mechanism", "unknown"),
            maia_rank=meta_w.get("maia_rank"),
            maia_top5=json.dumps(meta_w.get("maia_top5", [])),
            notes=meta_w.get("notes"),
        )

        mechanisms_white.append(meta_w.get("mechanism", "unknown"))
        move_list.append(san_w)

        # Sync black's board
        gs_black.board = gs_white.board.copy()
        gs_black.ply = gs_white.ply
        gs_black._current_eval_cp = gs_white._current_eval_cp

        if verbose and full_move <= 5:
            print(f"    {full_move}. {san_w}", end="")

        # === BLACK's turn ===
        if gs_black.board.is_game_over():
            if verbose and full_move <= 5:
                print()
            break

        fen_before = gs_black.board.fen()
        eval_before = black_bot.evaluate_position(gs_black)

        move_b, meta_b = black_bot.select_move(gs_black)
        san_b = gs_black.board.san(move_b)
        record_b = gs_black.push_move(move_b, eval_before=eval_before)

        # Get eval after
        eval_after = black_bot.evaluate_position(gs_black)
        record_b.eval_after = eval_after

        # Compute cp_loss (from black's perspective)
        cp_loss = None
        if eval_before is not None and eval_after is not None:
            cp_loss = max(0, eval_after - eval_before)
        was_blunder = cp_loss is not None and cp_loss > 100
        if was_blunder:
            blunders_black += 1

        # Log black's move
        logger.log_move(
            game_id=game_id,
            move_number=gs_black.move_number,
            ply=gs_black.ply,
            player=f"stonefish_{black_elo}",
            fen_before=fen_before,
            move_uci=move_b.uci(),
            move_san=san_b,
            piece_moved=record_b.piece_name,
            is_capture=record_b.is_capture,
            is_check=record_b.is_check,
            eval_before=eval_before,
            eval_after=eval_after,
            cp_loss=cp_loss,
            was_blunder=was_blunder,
            mechanism=meta_b.get("mechanism", "unknown"),
            maia_rank=meta_b.get("maia_rank"),
            maia_top5=json.dumps(meta_b.get("maia_top5", [])),
            notes=meta_b.get("notes"),
        )

        mechanisms_black.append(meta_b.get("mechanism", "unknown"))
        move_list.append(san_b)

        # Sync white's board
        gs_white.board = gs_black.board.copy()
        gs_white.ply = gs_black.ply
        gs_white._current_eval_cp = gs_black._current_eval_cp

        if verbose and full_move <= 5:
            print(f" {san_b}")

    # --- Game over ---
    elapsed = time.time() - game_start
    board = gs_white.board
    result = board.result() if board.is_game_over() else "1/2-1/2"

    # Determine termination reason
    if board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material():
        termination = "insufficient_material"
    elif board.is_fifty_moves():
        termination = "fifty_moves"
    elif board.is_repetition():
        termination = "repetition"
    elif not board.is_game_over():
        termination = "max_moves"
        result = "1/2-1/2"
    else:
        termination = "unknown"

    total_moves = len(move_list)

    logger.end_game(game_id, result=result, total_moves=total_moves // 2)

    # Mechanism breakdown
    mech_counts_w = {}
    for m in mechanisms_white:
        mech_counts_w[m] = mech_counts_w.get(m, 0) + 1
    mech_counts_b = {}
    for m in mechanisms_black:
        mech_counts_b[m] = mech_counts_b.get(m, 0) + 1

    return {
        "game_id": game_id,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "termination": termination,
        "total_plies": total_moves,
        "elapsed_seconds": round(elapsed, 1),
        "blunders_white": blunders_white,
        "blunders_black": blunders_black,
        "mechanisms_white": mech_counts_w,
        "mechanisms_black": mech_counts_b,
    }


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    pairings: List[Tuple[int, int]],
    games_per_pairing: int,
    verbose: bool = True,
) -> Dict:
    """Run a round-robin tournament between bot ELO tiers.

    Args:
        pairings: List of (elo_a, elo_b) tuples to play.
        games_per_pairing: Number of games per pairing (alternating colors).
        verbose: Print progress.

    Returns:
        Tournament results dict.
    """
    all_elos = sorted(set(e for pair in pairings for e in pair))

    print()
    print("=" * 60)
    print("   STONEFISH TOURNAMENT")
    print("=" * 60)
    print(f"   Bots: {', '.join(str(e) for e in all_elos)}")
    print(f"   Games per pairing: {games_per_pairing}")
    print(f"   Total games: {len(pairings) * games_per_pairing}")
    print("=" * 60)
    print()

    # --- Initialize shared engines ---
    print("  Starting engines...")
    maia = MaiaEngine()
    maia.start()
    print("    Maia ready")

    stockfish = StockfishEngine(depth=12, threads=2, hash_mb=256)
    stockfish.start()
    print("    Stockfish ready")

    # --- Create bots for each ELO ---
    bots = {}
    for elo in all_elos:
        bot = StonefishBot(
            elo_target=elo,
            maia_engine=maia,
            stockfish_engine=stockfish,
        )
        bot._engines_started = True  # Engines managed externally
        bots[elo] = bot
        print(f"    Bot {elo} ready (params: {bot.params['label']})")

    # --- Logger ---
    logger = GameLogger()
    logger.open()
    print("    Logger ready")
    print()

    # --- Run matches ---
    results = {
        "tournament_start": datetime.now().isoformat(),
        "games_per_pairing": games_per_pairing,
        "pairings": [],
        "standings": {},
    }

    # Initialize standings
    for elo in all_elos:
        results["standings"][str(elo)] = {
            "wins": 0, "draws": 0, "losses": 0, "score": 0.0,
            "total_blunders": 0, "games_played": 0,
        }

    total_games = len(pairings) * games_per_pairing
    game_num = 0

    for elo_a, elo_b in pairings:
        pairing_results = {
            "elo_a": elo_a,
            "elo_b": elo_b,
            "games": [],
            "a_wins": 0,
            "b_wins": 0,
            "draws": 0,
        }

        print(f"  === Stonefish {elo_a} vs Stonefish {elo_b} ===")

        for game_idx in range(games_per_pairing):
            game_num += 1
            # Alternate colors
            if game_idx % 2 == 0:
                white_elo, black_elo = elo_a, elo_b
            else:
                white_elo, black_elo = elo_b, elo_a

            white_bot = bots[white_elo]
            black_bot = bots[black_elo]

            # Reset blunder carry-forward for each game
            white_bot._blunder_carry_forward = False
            black_bot._blunder_carry_forward = False

            if verbose:
                print(f"  Game {game_num}/{total_games}: "
                      f"Stonefish {white_elo} (W) vs Stonefish {black_elo} (B)")

            game_result = play_game(
                white_bot=white_bot,
                black_bot=black_bot,
                white_elo=white_elo,
                black_elo=black_elo,
                logger=logger,
                verbose=verbose,
            )

            pairing_results["games"].append(game_result)

            # Track pairing results
            if game_result["result"] == "1-0":
                # White wins
                if white_elo == elo_a:
                    pairing_results["a_wins"] += 1
                else:
                    pairing_results["b_wins"] += 1
            elif game_result["result"] == "0-1":
                # Black wins
                if black_elo == elo_a:
                    pairing_results["a_wins"] += 1
                else:
                    pairing_results["b_wins"] += 1
            else:
                pairing_results["draws"] += 1

            # Update standings
            if game_result["result"] == "1-0":
                results["standings"][str(white_elo)]["wins"] += 1
                results["standings"][str(white_elo)]["score"] += 1.0
                results["standings"][str(black_elo)]["losses"] += 1
            elif game_result["result"] == "0-1":
                results["standings"][str(black_elo)]["wins"] += 1
                results["standings"][str(black_elo)]["score"] += 1.0
                results["standings"][str(white_elo)]["losses"] += 1
            else:
                results["standings"][str(white_elo)]["draws"] += 1
                results["standings"][str(white_elo)]["score"] += 0.5
                results["standings"][str(black_elo)]["draws"] += 1
                results["standings"][str(black_elo)]["score"] += 0.5

            results["standings"][str(white_elo)]["total_blunders"] += game_result["blunders_white"]
            results["standings"][str(black_elo)]["total_blunders"] += game_result["blunders_black"]
            results["standings"][str(white_elo)]["games_played"] += 1
            results["standings"][str(black_elo)]["games_played"] += 1

            # Print result
            emoji = {"1-0": "W", "0-1": "B", "1/2-1/2": "D", "*": "?"}
            r = game_result["result"]
            plies = game_result["total_plies"]
            t = game_result["elapsed_seconds"]
            term = game_result["termination"]
            bw = game_result["blunders_white"]
            bb = game_result["blunders_black"]
            print(f"    -> {r} ({emoji.get(r, '?')}) in {plies} plies, "
                  f"{t}s [{term}] blunders: W={bw} B={bb}")

        # Print pairing summary
        print(f"  Pairing result: {elo_a} {pairing_results['a_wins']}W "
              f"{pairing_results['draws']}D {pairing_results['b_wins']}L "
              f"(vs {elo_b})")
        print()

        results["pairings"].append(pairing_results)

    results["tournament_end"] = datetime.now().isoformat()

    # --- Print final standings ---
    print("=" * 60)
    print("   FINAL STANDINGS")
    print("=" * 60)
    print(f"  {'Bot':>12}  {'W':>3}  {'D':>3}  {'L':>3}  {'Score':>6}  {'Blunders':>8}  {'Avg Blunders':>12}")
    print(f"  {'-'*12}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*12}")

    for elo in sorted(results["standings"].keys(), key=lambda x: results["standings"][x]["score"], reverse=True):
        s = results["standings"][elo]
        games = s["games_played"]
        avg_blunders = s["total_blunders"] / games if games > 0 else 0
        print(f"  Stonefish {elo:>4}  {s['wins']:>3}  {s['draws']:>3}  {s['losses']:>3}  "
              f"{s['score']:>5.1f}  {s['total_blunders']:>8}  {avg_blunders:>11.1f}")

    print("=" * 60)

    # --- Shutdown ---
    logger.close()
    stockfish.stop()
    maia.stop()
    print("  Engines stopped. Database: data/stonefish_games.db")

    # --- Save JSON results ---
    out_path = Path("data") / "tournament_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stonefish Tournament -- round-robin bot vs bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tournament.py                        # default: 10 games each\n"
            "  python tournament.py --games 20             # 20 games per pairing\n"
            "  python tournament.py --pairings 500,700     # only 500 vs 700\n"
        ),
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Number of games per pairing (default: 10)",
    )
    parser.add_argument(
        "--pairings", type=str, default=None,
        help="Comma-separated ELO pairings (e.g. '500,700' or '500,700;700,900'). "
             "Default: full round-robin of 500, 700, 900.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output (no per-move details)",
    )
    args = parser.parse_args()

    if args.pairings:
        pairings = []
        for p in args.pairings.split(";"):
            parts = p.strip().split(",")
            if len(parts) == 2:
                pairings.append((int(parts[0]), int(parts[1])))
    else:
        # Full round-robin: 500 vs 700, 500 vs 900, 700 vs 900
        pairings = [(500, 700), (500, 900), (700, 900)]

    run_tournament(
        pairings=pairings,
        games_per_pairing=args.games,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
