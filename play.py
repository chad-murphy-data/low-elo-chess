#!/usr/bin/env python3
"""
Stonefish Chess Bot -- Play a game from the command line.

Usage:
    python play.py                          # 500 ELO bot, random color
    python play.py --elo 700 --color white  # play as white vs 700 bot
    python play.py --elo 900 --color black  # play as black vs 900 bot
"""

import argparse
import json
import random
import sys
from typing import Optional

import chess

from src.bot import StonefishBot
from src.game_state import GameState
from src.game_logger import GameLogger


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _can_use_unicode() -> bool:
    """Check if the terminal supports Unicode chess pieces."""
    try:
        "\u2654".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def display_board(board: chess.Board, perspective: chess.Color) -> None:
    """Print the board from the given perspective.

    Uses Unicode chess pieces if the terminal supports them,
    otherwise falls back to ASCII letters (K, Q, R, B, N, P / k, q, ...).
    """
    use_unicode = _can_use_unicode()

    if perspective == chess.WHITE:
        ranks = range(7, -1, -1)   # 8 down to 1
    else:
        ranks = range(0, 8)         # 1 up to 8

    files = range(0, 8) if perspective == chess.WHITE else range(7, -1, -1)

    unicode_symbols = {
        "R": "\u2656", "N": "\u2658", "B": "\u2657", "Q": "\u2655",
        "K": "\u2654", "P": "\u2659",
        "r": "\u265c", "n": "\u265e", "b": "\u265d", "q": "\u265b",
        "k": "\u265a", "p": "\u265f",
    }

    print()
    for rank in ranks:
        row = f"  {rank + 1} "
        for file_idx in files:
            sq = chess.square(file_idx, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += " . "
            else:
                sym = piece.symbol()
                if use_unicode:
                    sym = unicode_symbols.get(sym, sym)
                row += f" {sym} "
        print(row)

    # File labels
    file_labels = "    "
    for file_idx in files:
        file_labels += f" {chr(ord('a') + file_idx)} "
    print(file_labels)
    print()


def format_eval(eval_cp: Optional[int], bot_color: chess.Color) -> str:
    """Format eval for display from the bot's perspective."""
    if eval_cp is None:
        return "?"

    if bot_color == chess.WHITE:
        bot_eval = eval_cp
    else:
        bot_eval = -eval_cp

    if abs(eval_cp) >= 9000:
        if bot_eval > 0:
            return "Mate (bot winning)"
        else:
            return "Mate (bot losing)"

    pawns = bot_eval / 100.0
    if pawns > 0.05:
        return f"+{pawns:.1f} (bot ahead)"
    elif pawns < -0.05:
        return f"{pawns:.1f} (bot behind)"
    else:
        return "0.0 (equal)"


def format_eval_bar(eval_cp: Optional[int], width: int = 30) -> str:
    """Visual eval bar: [####......] style (ASCII-safe)."""
    if eval_cp is None:
        return ""

    # Clamp to [-500, +500] for display
    clamped = max(-500, min(500, eval_cp))
    # Map to [0, width]: 0 = full black advantage, width = full white advantage
    pos = int((clamped + 500) / 1000 * width)
    pos = max(0, min(width, pos))

    bar = "#" * pos + "." * (width - pos)
    return f"  [{bar}]"


# ---------------------------------------------------------------------------
# Move parsing
# ---------------------------------------------------------------------------

def parse_human_move(board: chess.Board, move_str: str) -> chess.Move:
    """Parse a human-entered move in UCI or SAN notation.

    Accepts: "e2e4", "e4", "Nf3", "O-O", "Qxd5+", "e7e8q", etc.

    Raises:
        ValueError: If the move is invalid or illegal.
    """
    move_str = move_str.strip()

    # Try UCI first (e.g., "e2e4", "e7e8q")
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except (chess.InvalidMoveError, ValueError):
        pass

    # Try SAN (e.g., "e4", "Nf3", "O-O", "Qxd5+")
    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move
    except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
        pass

    raise ValueError(
        f"Invalid or illegal move: '{move_str}'. "
        f"Use UCI (e2e4) or SAN (e4, Nf3, O-O) notation."
    )


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(elo: int, human_color: chess.Color) -> None:
    """Run a single game between human and bot."""
    bot_color = not human_color
    bot_color_str = "white" if bot_color == chess.WHITE else "black"
    human_color_str = "white" if human_color == chess.WHITE else "black"

    print()
    print("=" * 55)
    print(f"   STONEFISH {elo} vs Human")
    print(f"   Bot plays:  {bot_color_str}")
    print(f"   You play:   {human_color_str}")
    print("=" * 55)
    print()
    print("  Enter moves as UCI (e2e4) or SAN (e4, Nf3, O-O).")
    print("  Commands: 'resign', 'draw', 'undo', 'quit'")
    print()

    game_state = GameState(bot_color=bot_color)
    logger = GameLogger()
    logger.open()

    bot = StonefishBot(elo_target=elo)

    print("  Starting engines...")
    bot.start_engines()
    print("  Engines ready.\n")

    game_id = logger.start_game(
        bot_elo=elo,
        bot_color=bot_color_str,
        opponent_type="human_cli",
    )

    try:
        # Initial eval
        eval_cp = bot.evaluate_position(game_state)
        display_board(game_state.board, human_color)
        print(f"  Eval: {format_eval(eval_cp, bot_color)}")
        print(format_eval_bar(eval_cp))

        while not game_state.is_game_over:

            if game_state.is_bot_turn:
                # ===== Bot's turn =====
                print(f"  Stonefish {elo} is thinking...")
                fen_before = game_state.board.fen()
                eval_before = game_state.get_eval()

                move, metadata = bot.select_move(game_state)
                san = game_state.board.san(move)

                # Push move
                record = game_state.push_move(move, eval_before=eval_before)

                # Eval after
                eval_after = bot.evaluate_position(game_state)
                record.eval_after = eval_after

                # Compute cp_loss
                cp_loss = None
                if eval_before is not None and eval_after is not None:
                    if bot_color == chess.WHITE:
                        cp_loss = max(0, eval_before - eval_after)
                    else:
                        cp_loss = max(0, eval_after - eval_before)

                was_blunder = cp_loss is not None and cp_loss > 100

                # Log
                logger.log_move(
                    game_id=game_id,
                    move_number=game_state.move_number,
                    ply=game_state.ply,
                    player="bot",
                    fen_before=fen_before,
                    move_uci=move.uci(),
                    move_san=san,
                    piece_moved=record.piece_name,
                    is_capture=record.is_capture,
                    is_check=record.is_check,
                    eval_before=eval_before,
                    eval_after=eval_after,
                    cp_loss=cp_loss,
                    was_blunder=was_blunder,
                    mechanism=metadata["mechanism"],
                    maia_rank=metadata.get("maia_rank"),
                    maia_top5=json.dumps(metadata.get("maia_top5", [])),
                    notes=metadata.get("notes"),
                )

                # Display
                capture_str = " (capture)" if record.is_capture else ""
                check_str = "+" if record.is_check else ""
                print(f"\n  Stonefish plays: {san}{check_str}{capture_str}")
                if metadata.get("maia_rank"):
                    print(
                        f"    Maia rank {metadata['maia_rank']} / "
                        f"{bot.params['maia_sampling']['strategy']}"
                    )
                if was_blunder and cp_loss is not None:
                    print(f"    [BLUNDER: {cp_loss} cp loss]")

                display_board(game_state.board, human_color)
                print(f"  Eval: {format_eval(eval_after, bot_color)}")
                print(format_eval_bar(eval_after))

            else:
                # ===== Human's turn =====
                while True:
                    try:
                        raw = input("  Your move: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        raw = "quit"

                    if not raw:
                        continue

                    cmd = raw.lower()
                    if cmd == "quit":
                        print("\n  Exiting game.")
                        logger.end_game(
                            game_id, result="*",
                            total_moves=game_state.move_number,
                        )
                        return

                    if cmd == "resign":
                        result = "1-0" if bot_color == chess.WHITE else "0-1"
                        print(f"\n  You resigned. Result: {result}")
                        logger.end_game(
                            game_id, result=result,
                            total_moves=game_state.move_number,
                        )
                        return

                    if cmd == "draw":
                        result = "1/2-1/2"
                        print(f"\n  Draw agreed. Result: {result}")
                        logger.end_game(
                            game_id, result=result,
                            total_moves=game_state.move_number,
                        )
                        return

                    if cmd == "undo":
                        # Undo last two half-moves (human + bot)
                        if len(game_state.move_history) >= 2:
                            game_state.board.pop()
                            game_state.board.pop()
                            game_state.move_history.pop()
                            game_state.move_history.pop()
                            game_state.ply -= 2
                            eval_cp = bot.evaluate_position(game_state)
                            print("  Undid last move pair.")
                            display_board(game_state.board, human_color)
                            print(f"  Eval: {format_eval(eval_cp, bot_color)}")
                            print(format_eval_bar(eval_cp))
                        else:
                            print("  Nothing to undo.")
                        continue

                    if cmd == "moves":
                        legal = [
                            game_state.board.san(m)
                            for m in game_state.board.legal_moves
                        ]
                        print(f"  Legal moves: {', '.join(legal)}")
                        continue

                    if cmd == "fen":
                        print(f"  {game_state.board.fen()}")
                        continue

                    # Try to parse as a chess move
                    try:
                        move = parse_human_move(game_state.board, raw)
                        break
                    except ValueError as e:
                        print(f"  {e}")

                # Execute human move
                fen_before = game_state.board.fen()
                eval_before = game_state.get_eval()
                san = game_state.board.san(move)

                record = game_state.push_move(
                    move, eval_before=eval_before
                )

                eval_after = bot.evaluate_position(game_state)
                record.eval_after = eval_after

                # Compute cp_loss
                cp_loss = None
                if eval_before is not None and eval_after is not None:
                    if human_color == chess.WHITE:
                        cp_loss = max(0, eval_before - eval_after)
                    else:
                        cp_loss = max(0, eval_after - eval_before)

                was_blunder = cp_loss is not None and cp_loss > 100

                # Log
                logger.log_move(
                    game_id=game_id,
                    move_number=game_state.move_number,
                    ply=game_state.ply,
                    player="human",
                    fen_before=fen_before,
                    move_uci=move.uci(),
                    move_san=san,
                    piece_moved=record.piece_name,
                    is_capture=record.is_capture,
                    is_check=record.is_check,
                    eval_before=eval_before,
                    eval_after=eval_after,
                    cp_loss=cp_loss,
                    was_blunder=was_blunder,
                    mechanism="human_input",
                )

                # Display
                display_board(game_state.board, human_color)
                print(f"  Eval: {format_eval(eval_after, bot_color)}")
                print(format_eval_bar(eval_after))

        # ===== Game over =====
        result = game_state.game_result
        print()
        print("=" * 55)
        print(f"   Game Over: {result}")

        if game_state.board.is_checkmate():
            loser = "Black" if game_state.board.turn == chess.BLACK else "White"
            print(f"   Checkmate! {loser} is mated.")
        elif game_state.board.is_stalemate():
            print("   Stalemate!")
        elif game_state.board.is_insufficient_material():
            print("   Draw — insufficient material.")
        elif game_state.board.is_fifty_moves():
            print("   Draw — 50-move rule.")
        elif game_state.board.is_repetition():
            print("   Draw — threefold repetition.")

        print("=" * 55)

        logger.end_game(
            game_id, result=result, total_moves=game_state.move_number
        )
        print(f"\n  Game logged: {game_id}")
        print(f"  Database: data/stonefish_games.db")

    except KeyboardInterrupt:
        print("\n\n  Game interrupted.")
        logger.end_game(
            game_id, result="*", total_moves=game_state.move_number
        )

    finally:
        bot.stop_engines()
        logger.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play against the Stonefish chess bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python play.py                          # 500 ELO, random color\n"
            "  python play.py --elo 700 --color white  # you play white vs 700\n"
            "  python play.py --elo 900 --color black  # you play black vs 900\n"
        ),
    )
    parser.add_argument(
        "--elo", type=int, default=500, choices=[500, 700, 900],
        help="Bot target ELO (default: 500)",
    )
    parser.add_argument(
        "--color", default="random", choices=["white", "black", "random"],
        help="Your color (default: random)",
    )
    args = parser.parse_args()

    if args.color == "random":
        human_color = random.choice([chess.WHITE, chess.BLACK])
    elif args.color == "white":
        human_color = chess.WHITE
    else:
        human_color = chess.BLACK

    play_game(elo=args.elo, human_color=human_color)


if __name__ == "__main__":
    main()
