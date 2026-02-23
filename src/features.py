"""
Feature engineering: parse PGN games into a flat CSV of per-move features.

Each row = one move. Computes position features using python-chess,
extracts eval/clock from Lichess PGN comments, and tracks piece recency.
"""

import json
import os
import re
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

# Standard piece values for material balance and SEE
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# For SEE ordering: king is last (never voluntarily enters exchange)
SEE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10000,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


# ---------------------------------------------------------------------------
# PGN comment parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Board-state helpers
# ---------------------------------------------------------------------------

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


def chebyshev_distance(from_sq, to_sq):
    """Chessboard distance between two squares."""
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)
    return max(abs(to_file - from_file), abs(to_rank - from_rank))


def is_backwards_knight_move(move, moving_color):
    """Check if a knight move goes 'backwards' toward the player's back rank.

    For White: moving to a lower rank. For Black: moving to a higher rank.
    Returns True/False. Non-knight moves always return False.
    """
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if moving_color == chess.WHITE:
        return to_rank < from_rank
    else:
        return to_rank > from_rank


def _attacked_pieces(board, color):
    """Return set of squares where `color`'s pieces are attacked by opponent.

    Excludes kings.
    """
    opponent = not color
    attacked = set()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(opponent, sq):
            attacked.add(sq)
    return attacked


# ---------------------------------------------------------------------------
# Static Exchange Evaluation (SEE)
# ---------------------------------------------------------------------------

def see(board, square):
    """Static Exchange Evaluation on a given square.

    Simulates the full capture-recapture chain using
    least-valuable-attacker-first logic.

    Args:
        board: chess.Board (will be modified — pass a copy!)
        square: the destination square to evaluate

    Returns:
        Net material gain from the opponent's perspective (float).
        None if SEE <= 0 (the exchange favors the piece's owner or is neutral).
    """
    piece = board.piece_at(square)
    if piece is None:
        return None

    piece_owner = piece.color
    side_to_capture = not piece_owner  # opponent captures first

    # Build the gain list using negamax
    gain = [SEE_PIECE_VALUES.get(piece.piece_type, 0)]
    current_piece_value = gain[0]  # value of piece currently on the square

    side = side_to_capture
    depth = 0

    while True:
        # Find least valuable attacker from 'side'
        attackers = board.attackers(side, square)
        if not attackers:
            break

        # Pick the least valuable attacker
        min_val = 999999
        min_sq = None
        for sq in attackers:
            p = board.piece_at(sq)
            if p is not None:
                v = SEE_PIECE_VALUES.get(p.piece_type, 0)
                if v < min_val:
                    min_val = v
                    min_sq = sq

        if min_sq is None:
            break

        # If the attacker is a king, only allow capture if opponent has
        # no more attackers (otherwise king walks into danger)
        attacker_piece = board.piece_at(min_sq)
        if attacker_piece.piece_type == chess.KING:
            opponent_of_side = not side
            if board.attackers(opponent_of_side, square):
                break  # King can't safely capture

        depth += 1
        gain.append(current_piece_value - gain[depth - 1])
        current_piece_value = min_val  # the attacker is now on the square

        # Remove the attacker from the board (this enables x-ray discovery)
        board.remove_piece_at(min_sq)

        # Switch sides
        side = not side

    # Negamax unwind
    for d in range(depth - 1, 0, -1):
        gain[d - 1] = -max(-gain[d - 1], gain[d])

    # gain[0] is now the net gain from the first capturer's perspective
    # (i.e., the opponent of the piece owner)
    result = gain[0]
    if result <= 0:
        return None
    return float(result)


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def detect_patterns(moving_color, player_move_history, move, moved_piece, board, is_capture):
    """Detect known low-ELO attack patterns from the current player's move history.

    Returns:
        (is_executing_pattern: bool, pattern_type: str or None)
    """
    history = player_move_history[moving_color]
    player_move_num = len(history)  # 1-indexed: how many moves this player has made

    # --- Pattern 1: scholars_mate (White only, moves 1-3) ---
    if moving_color == chess.WHITE and player_move_num <= 3:
        if _check_scholars_mate(history, player_move_num):
            return True, "scholars_mate"

    # --- Pattern 2: early_f7_attack (either color, within first 6 moves) ---
    if player_move_num <= 6:
        if moved_piece and moved_piece.piece_type in (chess.BISHOP, chess.QUEEN):
            if _check_early_f7_attack(moving_color, move, board):
                return True, "early_f7_attack"

    # --- Pattern 3: early_queen_sortie (either color, before move 6, not capture) ---
    if player_move_num < 6 and not is_capture:
        if moved_piece and moved_piece.piece_type == chess.QUEEN:
            if _check_early_queen_sortie(moving_color, move):
                return True, "early_queen_sortie"

    return False, None


def _check_scholars_mate(history, player_move_num):
    """Check if White's moves 1-3 match the Scholar's Mate prefix: e4, Bc4, Qh5."""
    if player_move_num >= 1:
        m1_move, m1_piece = history[0]
        if not (m1_piece == chess.PAWN and m1_move.to_square == chess.E4):
            return False
    if player_move_num >= 2:
        m2_move, m2_piece = history[1]
        if not (m2_piece == chess.BISHOP and m2_move.to_square == chess.C4):
            return False
    if player_move_num >= 3:
        m3_move, m3_piece = history[2]
        if not (m3_piece == chess.QUEEN and m3_move.to_square == chess.H5):
            return False
    return True


def _check_early_f7_attack(moving_color, move, board):
    """Check if a bishop/queen move creates a two-piece attack on f7 (white) or f2 (black)."""
    target_sq = chess.F7 if moving_color == chess.WHITE else chess.F2
    attackers = board.attackers(moving_color, target_sq)
    if len(attackers) < 2:
        return False
    # The moved piece must be among the attackers of the target square
    if move.to_square not in attackers:
        # The piece might attack the target from its new square even if
        # it's not ON the target square — board.attackers handles this.
        # If it's still not in attackers, the move doesn't contribute.
        return False
    return True


def _check_early_queen_sortie(moving_color, move):
    """Check if queen moved to opponent's half of the board."""
    to_rank = chess.square_rank(move.to_square)
    if moving_color == chess.WHITE:
        return to_rank >= 4  # ranks 5-8 (0-indexed: 4-7)
    else:
        return to_rank <= 3  # ranks 1-4 (0-indexed: 0-3)


# ---------------------------------------------------------------------------
# Per-game feature extraction
# ---------------------------------------------------------------------------

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

    # Track new attacks created by each side's last move (for two-move attack detection)
    prev_new_attacks = {chess.WHITE: set(), chess.BLACK: set()}

    # Track per-player move history for pattern detection
    # Each entry: (move, piece_type)
    player_move_history = {chess.WHITE: [], chess.BLACK: []}

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

        # Piece being moved
        moved_piece = board.piece_at(move.from_square)
        piece_type_moved = PIECE_NAMES.get(
            moved_piece.piece_type, "unknown"
        ) if moved_piece else "unknown"
        piece_value_moved = PIECE_VALUES.get(
            moved_piece.piece_type, 0
        ) if moved_piece else 0
        is_capture = board.is_capture(move)
        move_dist = chebyshev_distance(move.from_square, move.to_square)

        # --- Tactical features (before push) ---
        # Backwards knight move
        is_backwards_knight = None
        if moved_piece and moved_piece.piece_type == chess.KNIGHT:
            is_backwards_knight = is_backwards_knight_move(move, moving_color)

        # Long bishop move (Chebyshev distance >= 4)
        is_long_bishop = None
        if moved_piece and moved_piece.piece_type == chess.BISHOP:
            is_long_bishop = move_dist >= 4

        # Two-move attack detection: track attacks BEFORE the move
        player_attacked_before = _attacked_pieces(board, moving_color)

        # Piece recency
        moves_since_last = None
        if move.from_square in piece_last_moved:
            moves_since_last = ply - piece_last_moved[move.from_square]
        else:
            moves_since_last = ply  # piece hasn't moved yet this game

        # --- Execute the move ---
        board.push(move)
        ply += 1

        # --- Features AFTER the move ---
        is_check_given = board.is_check()

        # Two-move attack detection (after push)
        player_attacked_after = _attacked_pieces(board, moving_color)
        new_attacks_on_player = player_attacked_after - player_attacked_before
        is_two_move_attack_target = False
        two_move_attack_value = 0
        if len(prev_new_attacks.get(moving_color, set())) > 0:
            is_two_move_attack_target = True
            for sq in prev_new_attacks.get(moving_color, set()):
                p = board.piece_at(sq)
                if p and p.color == moving_color:
                    two_move_attack_value = max(
                        two_move_attack_value,
                        PIECE_VALUES.get(p.piece_type, 0)
                    )
        prev_new_attacks[not moving_color] = new_attacks_on_player

        # --- NEW: piece_is_defended ---
        piece_is_defended = None
        if moved_piece and moved_piece.piece_type != chess.KING:
            defenders = board.attackers(moving_color, move.to_square)
            # Exclude king from defender count
            n_def_no_king = sum(
                1 for sq in defenders
                if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING
            )
            piece_is_defended = n_def_no_king > 0

        # --- NEW: hanging_piece_net_value (SEE) ---
        hanging_piece_net_value = None
        if piece_is_defended is False:
            hanging_piece_net_value = see(board.copy(), move.to_square)

        # --- NEW: distance_to_opponent_king ---
        opponent_king_sq = board.king(not moving_color)
        distance_to_opponent_king = chebyshev_distance(move.to_square, opponent_king_sq) if opponent_king_sq is not None else None

        # --- NEW: n_attackers, n_defenders ---
        n_attackers = len(board.attackers(not moving_color, move.to_square))
        n_defenders = len(board.attackers(moving_color, move.to_square))

        # --- NEW: pattern detection ---
        player_move_history[moving_color].append(
            (move, moved_piece.piece_type if moved_piece else None)
        )
        is_executing_pattern, pattern_type = detect_patterns(
            moving_color, player_move_history, move, moved_piece, board, is_capture
        )

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
        is_near_mate = False

        if eval_before is not None and eval_after is not None:
            if abs(eval_before) >= 9000 or abs(eval_after) >= 9000:
                is_near_mate = True

            if moving_color == chess.WHITE:
                cpl = eval_before - eval_after
            else:
                cpl = eval_after - eval_before

            centipawn_loss = max(0, cpl)

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
            "is_near_mate": is_near_mate,
            "num_legal_moves": num_legal_moves,
            "in_check": in_check,
            "piece_count": pc,
            "material_balance": mat_bal,
            "piece_type_moved": piece_type_moved,
            "piece_value_moved": piece_value_moved,
            "is_capture": is_capture,
            "is_check_given": is_check_given,
            "move_distance": move_dist,
            "moves_since_piece_last_moved": moves_since_last,
            "is_backwards_knight": is_backwards_knight,
            "is_long_bishop": is_long_bishop,
            "is_two_move_attack_target": is_two_move_attack_target,
            "two_move_attack_value": two_move_attack_value if is_two_move_attack_target else None,
            "piece_is_defended": piece_is_defended,
            "hanging_piece_net_value": hanging_piece_net_value,
            "distance_to_opponent_king": distance_to_opponent_king,
            "n_attackers": n_attackers,
            "n_defenders": n_defenders,
            "is_executing_pattern": is_executing_pattern,
            "pattern_type": pattern_type,
        }
        rows.append(row)

        # Update prev_eval for next iteration
        prev_eval_cp = current_eval_cp
        node = next_node

    return rows


def process_pgn_file(pgn_path):
    """Process all games in a PGN file into move-level features."""
    all_rows = []
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
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


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

COLUMNS = [
    "game_id", "move_number", "player_color", "player_elo", "opponent_elo",
    "clock_remaining", "time_control",
    "eval_before", "eval_after", "centipawn_loss", "is_near_mate",
    "num_legal_moves", "in_check", "piece_count", "material_balance",
    "piece_type_moved", "piece_value_moved",
    "is_capture", "is_check_given",
    "move_distance", "moves_since_piece_last_moved",
    "is_backwards_knight", "is_long_bishop",
    "is_two_move_attack_target", "two_move_attack_value",
    "piece_is_defended", "hanging_piece_net_value",
    "distance_to_opponent_king", "n_attackers", "n_defenders",
    "is_executing_pattern", "pattern_type",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir="data", output_file=None):
    """Process all PGN files in data/pgn/ and produce the moves CSV.

    Three-stage eval merge:
    1. Extract evals from PGN comments (Lichess server analysis)
    2. Join from existing CSV on (game_id, move_number, player_color)
    3. Fill remaining gaps from stockfish_evals.json

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

    # --- Type conversions ---
    bool_cols = [
        "is_near_mate", "in_check", "is_capture", "is_check_given",
        "is_backwards_knight", "is_long_bishop",
        "is_two_move_attack_target",
        "piece_is_defended", "is_executing_pattern",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    int_cols = [
        "move_number", "player_elo", "opponent_elo", "num_legal_moves",
        "piece_count", "move_distance", "piece_value_moved",
        "two_move_attack_value",
        "distance_to_opponent_king", "n_attackers", "n_defenders",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [
        "clock_remaining", "eval_before", "eval_after", "centipawn_loss",
        "material_balance", "moves_since_piece_last_moved",
        "hanging_piece_net_value",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Stage 2: Merge eval data from existing CSV ---
    existing_csv = Path(str(output_file))
    if existing_csv.exists():
        try:
            existing_df = pd.read_csv(
                existing_csv,
                usecols=["game_id", "move_number", "player_color",
                         "eval_before", "eval_after", "centipawn_loss", "is_near_mate"],
            )
            existing_df = existing_df.rename(columns={
                "eval_before": "_eval_before",
                "eval_after": "_eval_after",
                "centipawn_loss": "_centipawn_loss",
                "is_near_mate": "_is_near_mate",
            })
            df = df.merge(
                existing_df,
                on=["game_id", "move_number", "player_color"],
                how="left",
            )
            for col in ["eval_before", "eval_after", "centipawn_loss", "is_near_mate"]:
                src_col = f"_{col}"
                if src_col in df.columns:
                    mask = df[col].isna() & df[src_col].notna()
                    df.loc[mask, col] = df.loc[mask, src_col]
                    df.drop(columns=[src_col], inplace=True)

            filled_from_csv = df["eval_before"].notna().sum()
            print(f"Existing CSV merge: {filled_from_csv} rows with eval data")
        except Exception as e:
            print(f"Warning: could not merge from existing CSV: {e}")

    # --- Stage 3: Merge Stockfish evals from JSON ---
    sf_path = data_path / "stockfish_evals.json"
    if sf_path.exists():
        with open(sf_path) as f:
            sf_evals = json.load(f)

        missing_before = df["eval_before"].isna().sum()

        for gid, evals_list in sf_evals.items():
            mask = df["game_id"] == gid
            game_rows = df.loc[mask]
            if len(game_rows) == 0:
                continue

            for idx, row_idx in enumerate(game_rows.index):
                if idx < len(evals_list) - 1:
                    ev_before = evals_list[idx]
                    ev_after = evals_list[idx + 1]

                    if pd.isna(df.at[row_idx, "eval_before"]) and ev_before is not None:
                        df.at[row_idx, "eval_before"] = float(ev_before)
                    if pd.isna(df.at[row_idx, "eval_after"]) and ev_after is not None:
                        df.at[row_idx, "eval_after"] = float(ev_after)

        # Recompute centipawn_loss and is_near_mate for newly filled rows
        needs_recompute = (
            df["centipawn_loss"].isna()
            & df["eval_before"].notna()
            & df["eval_after"].notna()
        )
        if needs_recompute.any():
            recomp = df.loc[needs_recompute].copy()

            is_near_mate = (recomp["eval_before"].abs() >= 9000) | (
                recomp["eval_after"].abs() >= 9000
            )
            df.loc[needs_recompute, "is_near_mate"] = is_near_mate

            white_mask = recomp["player_color"] == "white"
            cpl = pd.Series(0.0, index=recomp.index)
            cpl[white_mask] = (
                recomp.loc[white_mask, "eval_before"]
                - recomp.loc[white_mask, "eval_after"]
            )
            cpl[~white_mask] = (
                recomp.loc[~white_mask, "eval_after"]
                - recomp.loc[~white_mask, "eval_before"]
            )
            cpl = cpl.clip(lower=0)
            df.loc[needs_recompute, "centipawn_loss"] = cpl

        missing_after = df["eval_before"].isna().sum()
        print(f"Stockfish evals: filled {missing_before - missing_after} / {missing_before} missing eval_before values")

    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file} ({len(df)} rows, {len(df.columns)} columns)")

    return df


if __name__ == "__main__":
    build_dataset()
