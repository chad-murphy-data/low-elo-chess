"""
Feature engineering: parse PGN games into a flat CSV of per-move features.

Each row = one move. Computes position features using python-chess,
extracts eval/clock from Lichess PGN comments, and tracks piece recency.
"""

import csv
import io
import os
import re
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

# Standard piece values for material balance
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def parse_eval(comment):
    """Extract centipawn eval from a Lichess PGN comment.

    Evals are always from White's perspective.
    Mate scores are mapped to ±10000 centipawns.
    """
    if not comment:
        return None
    mate_match = re.search(r"\[%eval #(-?\d+)\]", comment)
    if mate_match:
        mate_in = int(mate_match.group(1))
        return 10000 if mate_in > 0 else -10000
    eval_match = re.search(r"\[%eval (-?\d+\.?\d*)\]", comment)
    if eval_match:
        return float(eval_match.group(1)) * 100  # convert to centipawns
    return None


def parse_clock(comment):
    """Extract clock time remaining in seconds from a Lichess PGN comment."""
    if not comment:
        return None
    clk_match = re.search(r"\[%clk (\d+):(\d+):(\d+)\]", comment)
    if clk_match:
        h, m, s = (
            int(clk_match.group(1)),
            int(clk_match.group(2)),
            int(clk_match.group(3)),
        )
        return h * 3600 + m * 60 + s
    return None


def material_balance(board, perspective):
    """Compute material balance from perspective's point of view.

    Positive = perspective has more material.
    """
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == perspective:
            balance += val
        else:
            balance -= val
    return balance


def piece_count(board):
    """Total number of pieces on the board (including kings and pawns)."""
    return len(board.piece_map())


def get_hanging_pieces(board, color):
    """Return list of squares with hanging pieces of the given color.

    A piece is "hanging" if:
    - It's attacked by the opponent, AND
    - It's either undefended, or the cheapest attacker is worth less than it.

    Kings are excluded (can't be captured).
    """
    hanging = []
    opponent = not color
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != color:
            continue
        if piece.piece_type == chess.KING:
            continue
        if not board.is_attacked_by(opponent, square):
            continue
        defenders = board.attackers(color, square)
        if not defenders:
            hanging.append(square)
        else:
            attackers = board.attackers(opponent, square)
            piece_val = PIECE_VALUES[piece.piece_type]
            min_attacker_val = min(
                PIECE_VALUES.get(
                    board.piece_at(sq).piece_type, 0
                )
                for sq in attackers
                if board.piece_at(sq) is not None
            )
            if min_attacker_val < piece_val:
                hanging.append(square)
    return hanging


def chebyshev_distance(from_sq, to_sq):
    """Chessboard distance between two squares."""
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)
    return max(abs(to_file - from_file), abs(to_rank - from_rank))


def process_game(game, game_id=None):
    """Process a single chess.pgn.Game into a list of move-level feature dicts.

    Args:
        game: A chess.pgn.Game object with eval/clock comments.
        game_id: Optional game ID override.

    Returns:
        List of dicts, one per move (ply).
    """
    headers = game.headers
    if game_id is None:
        game_id = headers.get("Site", "").split("/")[-1]
        if not game_id:
            game_id = headers.get("Event", "unknown")

    white_elo_str = headers.get("WhiteElo", "")
    black_elo_str = headers.get("BlackElo", "")
    time_control = headers.get("TimeControl", "")

    try:
        white_elo = int(white_elo_str) if white_elo_str and white_elo_str != "?" else None
        black_elo = int(black_elo_str) if black_elo_str and black_elo_str != "?" else None
    except ValueError:
        white_elo = None
        black_elo = None

    # Skip games with missing ELO
    if white_elo is None or black_elo is None:
        return []

    board = game.board()
    rows = []

    # Track piece recency: square -> ply when a piece on that square last moved
    piece_last_moved = {}

    # Walk the main line
    node = game
    prev_eval_cp = None  # eval in centipawns (White's perspective) before this move
    ply = 0

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        comment = next_node.comment

        # Determine who is moving (before the move is pushed)
        moving_color = board.turn  # True = White, False = Black
        color_str = "white" if moving_color == chess.WHITE else "black"
        player_elo = white_elo if moving_color == chess.WHITE else black_elo
        opponent_elo = black_elo if moving_color == chess.WHITE else white_elo

        # --- Position features BEFORE the move ---
        num_legal_moves = board.legal_moves.count()
        in_check = board.is_check()
        pc = piece_count(board)
        mat_bal = material_balance(board, moving_color)

        # Hanging pieces before move
        player_hanging_before = get_hanging_pieces(board, moving_color)
        opponent_hanging_before = get_hanging_pieces(board, not moving_color)
        has_hanging_piece_before = len(player_hanging_before) > 0
        opponent_had_hanging_piece = len(opponent_hanging_before) > 0

        # Piece being moved
        moved_piece = board.piece_at(move.from_square)
        piece_type_moved = PIECE_NAMES.get(
            moved_piece.piece_type, "unknown"
        ) if moved_piece else "unknown"
        is_capture = board.is_capture(move)
        move_dist = chebyshev_distance(move.from_square, move.to_square)

        # Piece recency
        moves_since_last = None
        if move.from_square in piece_last_moved:
            moves_since_last = ply - piece_last_moved[move.from_square]
        else:
            moves_since_last = ply  # piece hasn't moved yet this game

        # Did the player capture the hanging piece?
        captured_hanging = False
        if opponent_had_hanging_piece and is_capture:
            if move.to_square in opponent_hanging_before:
                captured_hanging = True

        # Missed capture recency
        missed_capture_recency = None
        if opponent_had_hanging_piece and not captured_hanging:
            # Find the recency of the most recently moved hanging piece
            recencies = []
            for sq in opponent_hanging_before:
                if sq in piece_last_moved:
                    recencies.append(ply - piece_last_moved[sq])
                else:
                    recencies.append(ply)
            if recencies:
                missed_capture_recency = min(recencies)

        # --- Execute the move ---
        board.push(move)
        ply += 1

        # --- Features AFTER the move ---
        is_check_given = board.is_check()

        # Hanging pieces after move (from the moving player's perspective)
        player_hanging_after = get_hanging_pieces(board, moving_color)
        created_hanging_piece = len(player_hanging_after) > 0

        # Hung piece recency (if we created a hanging piece)
        hung_piece_recency = None
        if created_hanging_piece:
            recencies = []
            for sq in player_hanging_after:
                if sq in piece_last_moved:
                    recencies.append(ply - piece_last_moved[sq])
                else:
                    recencies.append(ply)
            if recencies:
                hung_piece_recency = min(recencies)

        # Update piece recency tracking
        piece_last_moved[move.to_square] = ply
        if move.from_square in piece_last_moved:
            del piece_last_moved[move.from_square]

        # --- Eval and clock from comment ---
        current_eval_cp = parse_eval(comment)
        clock_remaining = parse_clock(comment)

        # Compute centipawn loss
        eval_before = prev_eval_cp
        eval_after = current_eval_cp

        centipawn_loss = None
        is_blunder = None
        is_mistake = None
        is_inaccuracy = None
        is_near_mate = False

        if eval_before is not None and eval_after is not None:
            # Check for near-mate evals
            if abs(eval_before) >= 9000 or abs(eval_after) >= 9000:
                is_near_mate = True

            # Eval is from White's perspective. For the moving player:
            # If White moved: loss = eval_before - eval_after (positive = White lost)
            # If Black moved: loss = eval_after - eval_before (positive = Black lost,
            #   because higher eval = better for White = worse for Black)
            if moving_color == chess.WHITE:
                cpl = eval_before - eval_after
            else:
                cpl = eval_after - eval_before

            # CPL should be non-negative for a worsening move
            # (negative means the move improved the position)
            centipawn_loss = max(0, cpl)

            if not is_near_mate:
                is_blunder = centipawn_loss > 100
                is_mistake = centipawn_loss > 50
                is_inaccuracy = centipawn_loss > 25

        row = {
            "game_id": game_id,
            "move_number": ply,
            "player_color": color_str,
            "player_elo": player_elo,
            "opponent_elo": opponent_elo,
            "clock_remaining": clock_remaining,
            "time_control": time_control,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "centipawn_loss": centipawn_loss,
            "is_blunder": is_blunder,
            "is_mistake": is_mistake,
            "is_inaccuracy": is_inaccuracy,
            "is_near_mate": is_near_mate,
            "num_legal_moves": num_legal_moves,
            "in_check": in_check,
            "piece_count": pc,
            "material_balance": mat_bal,
            "has_hanging_piece_before": has_hanging_piece_before,
            "created_hanging_piece": created_hanging_piece,
            "opponent_had_hanging_piece": opponent_had_hanging_piece,
            "move_distance": move_dist,
            "piece_type_moved": piece_type_moved,
            "is_capture": is_capture,
            "is_check_given": is_check_given,
            "moves_since_piece_last_moved": moves_since_last,
            "hung_piece_recency": hung_piece_recency,
            "missed_capture_recency": missed_capture_recency,
        }
        rows.append(row)

        # Update prev_eval for next iteration
        prev_eval_cp = current_eval_cp
        node = next_node

    return rows


def process_pgn_file(pgn_path):
    """Process all games in a PGN file into move-level features."""
    all_rows = []
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Skip short games (< 10 moves = 20 ply)
            mainline_moves = list(game.mainline_moves())
            if len(mainline_moves) < 20:
                continue

            rows = process_game(game)
            all_rows.extend(rows)

    return all_rows


COLUMNS = [
    "game_id", "move_number", "player_color", "player_elo", "opponent_elo",
    "clock_remaining", "time_control",
    "eval_before", "eval_after", "centipawn_loss",
    "is_blunder", "is_mistake", "is_inaccuracy", "is_near_mate",
    "num_legal_moves", "in_check", "piece_count", "material_balance",
    "has_hanging_piece_before", "created_hanging_piece",
    "opponent_had_hanging_piece",
    "move_distance", "piece_type_moved", "is_capture", "is_check_given",
    "moves_since_piece_last_moved", "hung_piece_recency",
    "missed_capture_recency",
]


def build_dataset(data_dir="data", output_file=None):
    """Process all PGN files in data/pgn/ and produce the moves CSV.

    Args:
        data_dir: Root data directory containing pgn/ subdirectory.
        output_file: Path for output CSV. Defaults to data/moves_features.csv.

    Returns:
        pandas DataFrame of the full dataset.
    """
    data_path = Path(data_dir)
    pgn_dir = data_path / "pgn"

    if output_file is None:
        output_file = data_path / "moves_features.csv"

    pgn_files = sorted(pgn_dir.glob("*.pgn"))
    if not pgn_files:
        print("No PGN files found in", pgn_dir)
        return pd.DataFrame(columns=COLUMNS)

    print(f"Processing {len(pgn_files)} PGN files...")
    all_rows = []

    for i, pgn_file in enumerate(pgn_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(pgn_files)}] {pgn_file.name}")
        try:
            rows = process_pgn_file(pgn_file)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  Error processing {pgn_file.name}: {e}")
            continue

    print(f"Total move observations: {len(all_rows)}")

    df = pd.DataFrame(all_rows, columns=COLUMNS)

    # Type conversions
    bool_cols = [
        "is_blunder", "is_mistake", "is_inaccuracy", "is_near_mate",
        "in_check", "has_hanging_piece_before", "created_hanging_piece",
        "opponent_had_hanging_piece", "is_capture", "is_check_given",
    ]
    for col in bool_cols:
        df[col] = df[col].astype("boolean")  # nullable boolean

    int_cols = [
        "move_number", "player_elo", "opponent_elo", "num_legal_moves",
        "piece_count", "move_distance",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [
        "clock_remaining", "eval_before", "eval_after", "centipawn_loss",
        "material_balance", "moves_since_piece_last_moved",
        "hung_piece_recency", "missed_capture_recency",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")

    return df


if __name__ == "__main__":
    build_dataset()
