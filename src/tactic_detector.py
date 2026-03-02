"""
Stonefish Tactic Detector
=========================
Pattern detectors do geometry only — engine confirms whether it's real.

Four detectors: knight fork, general fork, trapped piece, discovered attack.
Each takes a board + color and returns candidate Tactics with estimated_gain=0.
The engine_validate_tactics() function filters candidates through Stockfish.

Stalemate warnings are handled separately via check_stalemate_warning().
"""

import chess
from dataclasses import dataclass, field
from typing import Optional


# Standard piece values for material calculations
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,  # Can be forked but has no "capture" value
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass
class Tactic:
    """A detected tactical opportunity."""
    tactic_type: str          # e.g. "knight_fork", "pin", "trapped_piece"
    description: str          # Human-readable description
    key_square: Optional[chess.Square] = None  # The square where the action happens
    attacker: Optional[chess.Square] = None    # Piece delivering the tactic
    targets: list = field(default_factory=list)  # Squares of targeted pieces
    estimated_gain: float = 0.0  # Set by engine validation (eval swing in pawns)
    move: Optional[chess.Move] = None  # The move that executes the tactic

    def __repr__(self):
        return f"<{self.tactic_type}: {self.description} (~{self.estimated_gain:+.1f})>"


def piece_name_at(board: chess.Board, square: chess.Square) -> str:
    """Get human-readable piece name at a square."""
    piece = board.piece_at(square)
    if piece is None:
        return "empty"
    return PIECE_NAMES.get(piece.piece_type, "unknown")


def square_name(square: chess.Square) -> str:
    """Get algebraic name of a square."""
    return chess.square_name(square)


# =============================================================================
# KNIGHT FORK DETECTOR (pure geometry)
# =============================================================================

def detect_knight_forks(board: chess.Board, color: chess.Color) -> list[Tactic]:
    """
    Detect available knight forks for the given color.

    Pure geometry: a knight can move to a square attacking 2+ enemy pieces
    worth >= bishop (3) or king. No gain estimation, no safety checks.
    """
    tactics = []
    opponent = not color

    for knight_sq in board.pieces(chess.KNIGHT, color):
        for move in board.legal_moves:
            if move.from_square != knight_sq:
                continue
            if board.piece_type_at(move.from_square) != chess.KNIGHT:
                continue

            fork_sq = move.to_square

            # Push move to check attacks from the landing square
            board.push(move)
            attacked_squares = board.attacks(fork_sq)

            targets = []
            for target_sq in attacked_squares:
                target_piece = board.piece_at(target_sq)
                if target_piece and target_piece.color == opponent:
                    t_val = PIECE_VALUES.get(target_piece.piece_type, 0)
                    if target_piece.piece_type == chess.KING or t_val >= 3:
                        targets.append(target_sq)

            board.pop()

            if len(targets) < 2:
                continue

            target_names = [f"{piece_name_at(board, t)} on {square_name(t)}" for t in targets[:3]]
            desc = f"Knight fork from {square_name(fork_sq)} attacking {', '.join(target_names)}"

            tactics.append(Tactic(
                tactic_type="knight_fork",
                description=desc,
                key_square=fork_sq,
                attacker=knight_sq,
                targets=list(targets),
                estimated_gain=0.0,
                move=move,
            ))

    return tactics


# =============================================================================
# GENERAL FORK DETECTOR — bishop, pawn, rook, queen (pure geometry)
# =============================================================================

def detect_general_forks(board: chess.Board, color: chess.Color) -> list[Tactic]:
    """
    Detect fork opportunities for bishops, pawns, queens, and rooks.

    Pure geometry: a piece moves to a square attacking 2+ enemy pieces
    worth >= bishop (3) or king. Knights handled separately.
    """
    tactics = []
    opponent = not color
    fork_piece_types = [chess.PAWN, chess.BISHOP, chess.ROOK, chess.QUEEN]

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.color != color:
            continue
        if piece.piece_type not in fork_piece_types:
            continue

        fork_sq = move.to_square
        piece_type = piece.piece_type

        # Push to check attacks from landing square
        board.push(move)
        attacked_squares = board.attacks(fork_sq)

        targets = []
        for t in attacked_squares:
            t_piece = board.piece_at(t)
            if t_piece is None or t_piece.color != opponent:
                continue
            t_val = PIECE_VALUES.get(t_piece.piece_type, 0)
            if t_piece.piece_type == chess.KING or t_val >= 3:
                targets.append(t)

        board.pop()

        if len(targets) < 2:
            continue

        piece_label = PIECE_NAMES.get(piece_type, "piece")
        target_names = [f"{piece_name_at(board, t)} on {square_name(t)}" for t in targets[:3]]
        desc = f"{piece_label.capitalize()} fork from {square_name(fork_sq)} attacking {', '.join(target_names)}"

        tactics.append(Tactic(
            tactic_type=f"{piece_label}_fork",
            description=desc,
            key_square=fork_sq,
            attacker=move.from_square,
            targets=list(targets),
            estimated_gain=0.0,
            move=move,
        ))

    return tactics


# =============================================================================
# TRAPPED PIECE DETECTOR (pure geometry)
# =============================================================================

def detect_trapped_pieces(board: chess.Board, color: chess.Color) -> list[Tactic]:
    """
    Detect opponent pieces that are trapped (no safe squares).

    Pure geometry: an opponent piece >= knight has zero safe escape squares
    and can't trade itself for equal value. No attacker/defender counting.
    """
    tactics = []
    opponent = not color

    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        for sq in board.pieces(piece_type, opponent):
            piece_value = PIECE_VALUES[piece_type]

            # Flip turn to opponent so we can enumerate their piece's moves
            need_flip = (board.turn == color)
            if need_flip:
                if board.is_check():
                    continue  # Can't null-move when in check
                board.push(chess.Move.null())

            safe_squares = 0
            can_trade_equal = False

            for move in board.legal_moves:
                if move.from_square != sq:
                    continue

                dest = move.to_square

                # Can it capture one of our pieces of equal+ value?
                captured = board.piece_at(dest)
                if captured and captured.color == color:
                    if PIECE_VALUES.get(captured.piece_type, 0) >= piece_value:
                        can_trade_equal = True
                        break

                # Is the destination safe?
                board.push(move)
                if not board.is_attacked_by(color, dest):
                    safe_squares += 1
                board.pop()

            if need_flip:
                board.pop()  # Undo null move

            if can_trade_equal:
                continue

            if safe_squares == 0:
                desc = (f"Trapped {piece_name_at(board, sq)} on {square_name(sq)} "
                       f"— no safe squares and can't trade for equal value")

                tactics.append(Tactic(
                    tactic_type="trapped_piece",
                    description=desc,
                    key_square=sq,
                    targets=[sq],
                    estimated_gain=0.0,
                ))

    return tactics


# =============================================================================
# DISCOVERED ATTACK DETECTOR (pure geometry)
# =============================================================================

def detect_discovered_attacks(board: chess.Board, color: chess.Color) -> list[Tactic]:
    """
    Detect discovered attack opportunities.

    Pure geometry: one of our pieces blocks a ray between our slider and
    an opponent piece. The blocker has a forcing move (check or capture)
    that unleashes the slider. Target must be worth >= bishop or be king.
    """
    tactics = []
    opponent = not color

    SLIDER_RAYS = {
        chess.BISHOP: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        chess.ROOK: [(0, 1), (0, -1), (1, 0), (-1, 0)],
        chess.QUEEN: [(-1, -1), (-1, 1), (1, -1), (1, 1),
                      (0, 1), (0, -1), (1, 0), (-1, 0)],
    }

    for slider_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for slider_sq in board.pieces(slider_type, color):
            rays = SLIDER_RAYS[slider_type]

            for dr, df in rays:
                blocker_sq = None
                target_sq = None

                r, f = chess.square_rank(slider_sq), chess.square_file(slider_sq)

                for step in range(1, 8):
                    nr, nf = r + dr * step, f + df * step
                    if not (0 <= nr <= 7 and 0 <= nf <= 7):
                        break

                    check_sq = chess.square(nf, nr)
                    piece = board.piece_at(check_sq)

                    if piece is None:
                        continue

                    if piece.color == color:
                        if blocker_sq is None:
                            blocker_sq = check_sq
                        else:
                            break  # Two of our pieces — no discovered attack
                    else:
                        if blocker_sq is not None:
                            target_sq = check_sq
                        break  # Enemy piece without blocker = normal attack

                if blocker_sq is None or target_sq is None:
                    continue

                target_piece = board.piece_at(target_sq)
                target_val = PIECE_VALUES.get(target_piece.piece_type, 0) if target_piece else 0

                # Target must be worth something
                if target_val < 3 and (target_piece and target_piece.piece_type != chess.KING):
                    continue

                # Find forcing blocker move (check or capture)
                best_blocker_move = None

                for move in board.legal_moves:
                    if move.from_square != blocker_sq:
                        continue

                    board.push(move)
                    gives_check = board.is_check()
                    board.pop()

                    captured = board.piece_at(move.to_square)
                    is_capture = captured is not None and captured.color == opponent

                    if gives_check or is_capture:
                        best_blocker_move = move
                        if gives_check:
                            break  # Discovered attack + check is the dream

                if best_blocker_move is None:
                    continue

                # Classify as discovered check or attack
                board.push(best_blocker_move)
                is_discovered_check = board.is_check()
                board.pop()

                ttype = "discovered_check" if is_discovered_check else "discovered_attack"

                blocker_piece = board.piece_at(blocker_sq)
                desc = (f"Discovered {'check' if is_discovered_check else 'attack'}: "
                       f"move {piece_name_at(board, blocker_sq)} from {square_name(blocker_sq)}, "
                       f"unleashing {piece_name_at(board, slider_sq)} on "
                       f"{piece_name_at(board, target_sq)} at {square_name(target_sq)}")

                tactics.append(Tactic(
                    tactic_type=ttype,
                    description=desc,
                    key_square=blocker_sq,
                    attacker=slider_sq,
                    targets=[target_sq],
                    estimated_gain=0.0,
                    move=best_blocker_move,
                ))

    return tactics


# =============================================================================
# MASTER DETECTOR — RUN ALL
# =============================================================================

ALL_DETECTORS = [
    ("knight_fork", detect_knight_forks),
    ("general_fork", detect_general_forks),
    ("trapped_piece", detect_trapped_pieces),
    ("discovered_attack", detect_discovered_attacks),
]


def detect_all_tactics(board: chess.Board, color: chess.Color) -> list[Tactic]:
    """Run all pattern detectors on a position. Returns unvalidated candidates."""
    all_tactics = []
    for name, detector in ALL_DETECTORS:
        try:
            results = detector(board, color)
            all_tactics.extend(results)
        except Exception as e:
            print(f"Warning: {name} detector failed: {e}")
    return all_tactics


# =============================================================================
# ENGINE VALIDATION LAYER
# =============================================================================

def engine_validate_tactics(
    tactics: list[Tactic],
    board: chess.Board,
    engine,  # StockfishEngine — duck-typed to avoid circular import
    color: chess.Color,
    min_swing_cp: int = 150,
) -> list[Tactic]:
    """Filter tactics through Stockfish evaluation.

    For each tactic with a move:
      1. Evaluate current position
      2. Push tactic move, get best response, evaluate resulting position
      3. Compute eval swing from tactic-player's perspective
      4. Keep if swing >= min_swing_cp

    Tactics without a move (e.g., trapped_piece) pass through unfiltered.
    Updates estimated_gain on surviving tactics to the eval swing in pawns.
    """
    if not tactics:
        return []

    # Get baseline eval once
    try:
        current_eval = engine.evaluate(board)
    except Exception:
        current_eval = None

    if current_eval is None:
        return tactics  # Can't validate — return all as fallback

    validated = []
    for t in tactics:
        if t.move is None:
            # No move to validate (trapped_piece) — pass through
            validated.append(t)
            continue

        if t.move not in board.legal_moves:
            continue  # Stale/invalid move

        try:
            board.push(t.move)

            if board.is_game_over():
                # Checkmate/stalemate — big swing, keep it
                if board.is_checkmate():
                    t.estimated_gain = 100.0  # Checkmate
                    validated.append(t)
                board.pop()
                continue

            # Get best response
            best_response = engine.get_best_move(board)
            if best_response:
                board.push(best_response)
                post_eval = engine.evaluate(board)
                board.pop()
            else:
                post_eval = engine.evaluate(board)

            board.pop()

            if post_eval is None:
                continue

            # Eval swing from tactic-player's perspective
            if color == chess.WHITE:
                swing = post_eval - current_eval
            else:
                swing = current_eval - post_eval

            if swing >= min_swing_cp:
                t.estimated_gain = swing / 100.0
                validated.append(t)

        except Exception as e:
            # Undo any pushed moves on error
            while board.move_stack and len(board.move_stack) > 0:
                try:
                    board.pop()
                except Exception:
                    break
            continue

    return validated


# =============================================================================
# STALEMATE WARNING SYSTEM
# =============================================================================

def check_stalemate_warning(board: chess.Board, color: chess.Color) -> Optional[dict]:
    """Check if the human (color) risks stalemating the opponent.

    Only runs in endgame: total pieces <= 7, or kings+pawns only.

    Buckets all legal moves into:
      - stalemating: results in immediate stalemate
      - winning: doesn't stalemate, doesn't hang material >= bishop
      - risky: hangs material or otherwise bad

    Returns a warning dict if stalemating is non-empty AND winning is non-empty.
    Returns None otherwise.
    """
    total_pieces = len(board.piece_map())
    opponent = not color

    # Check endgame conditions
    is_endgame = False
    if total_pieces <= 7:
        is_endgame = True
    else:
        # Check if only kings + pawns remain
        non_pawn_king = False
        for sq, piece in board.piece_map().items():
            if piece.piece_type not in (chess.KING, chess.PAWN):
                non_pawn_king = True
                break
        if not non_pawn_king:
            is_endgame = True

    if not is_endgame:
        return None

    # Must be color's turn (or about to be)
    if board.turn != color:
        return None

    stalemating = []
    winning = []
    risky = []

    for move in board.legal_moves:
        board.push(move)

        if board.is_stalemate():
            san = board.peek().__str__()  # Not reliable; get SAN before push
            board.pop()
            san = board.san(move) if move in board.legal_moves else move.uci()
            stalemating.append(move)
            continue

        # Check if this move hangs material >= bishop
        is_risky = False
        moved_piece = board.piece_at(move.to_square)
        if moved_piece:
            moved_val = PIECE_VALUES.get(moved_piece.piece_type, 0)
            if moved_val >= 3:
                # Is the destination attacked by opponent and undefended by us?
                if board.is_attacked_by(opponent, move.to_square):
                    defenders = board.attackers(color, move.to_square)
                    if not defenders:
                        is_risky = True

        # Check if opponent has a pawn about to promote
        promo_rank = 1 if opponent == chess.WHITE else 6  # 7th rank (0-indexed)
        for pawn_sq in board.pieces(chess.PAWN, opponent):
            if chess.square_rank(pawn_sq) == promo_rank:
                is_risky = True
                break

        board.pop()

        if is_risky:
            risky.append(move)
        else:
            winning.append(move)

    if stalemating and winning:
        avoid_str = ", ".join(board.san(m) for m in stalemating[:3])
        play_str = ", ".join(board.san(m) for m in winning[:3])
        desc = (f"Stalemate danger! Avoid: {avoid_str}. "
               f"Safe moves include: {play_str}")

        return {
            "type": "stalemate_danger",
            "description": desc,
            "gain": 0,
        }

    return None


# =============================================================================
# GAME ANALYZER — RUN DETECTORS ON A FULL GAME
# =============================================================================

def analyze_game(pgn_text: str, player_color: chess.Color = chess.WHITE,
                 engine=None, dedup: bool = True) -> dict:
    """
    Analyze a complete game for tactical opportunities.

    Args:
        pgn_text: PGN string of the game
        player_color: Which color the player was (tactics are detected for this color)
        engine: Optional StockfishEngine for validation. If None, returns raw patterns.
        dedup: If True, suppress tactics same as previous move (persistent tactics)

    Returns:
        Dict with move-by-move tactic detection results and summary stats
    """
    import chess.pgn
    import io

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return {"error": "Could not parse PGN"}

    board = game.board()
    moves = list(game.mainline_moves())

    results = {
        "total_moves": len(moves),
        "player_color": "white" if player_color == chess.WHITE else "black",
        "tactics_found": [],
        "summary": {},
    }

    tactic_counts = {}
    prev_tactic_keys = set()

    for i, move in enumerate(moves):
        move_num = i // 2 + 1
        is_player_turn = (board.turn == player_color)

        if is_player_turn:
            # Before the player moves, what tactics are available?
            raw_tactics = detect_all_tactics(board, player_color)

            # Engine-validate if engine available
            if engine is not None:
                tactics = engine_validate_tactics(
                    raw_tactics, board, engine, player_color
                )
            else:
                tactics = raw_tactics

            # Deduplicate
            if dedup:
                current_keys = set()
                deduped_tactics = []
                for t in tactics:
                    key = (t.tactic_type, tuple(sorted(t.targets)), t.key_square)
                    current_keys.add(key)
                    if key not in prev_tactic_keys:
                        deduped_tactics.append(t)
                prev_tactic_keys = current_keys
                tactics = deduped_tactics
            else:
                prev_tactic_keys = set()

            if tactics:
                player_found = []
                player_missed = []

                for tactic in tactics:
                    if tactic.move and tactic.move == move:
                        player_found.append(tactic)
                    else:
                        player_missed.append(tactic)

                move_result = {
                    "move_number": move_num,
                    "player_move": move.uci(),
                    "tactics_available": len(tactics),
                    "found": [repr(t) for t in player_found],
                    "missed": [repr(t) for t in player_missed],
                }
                results["tactics_found"].append(move_result)

                for t in tactics:
                    ttype = t.tactic_type
                    if ttype not in tactic_counts:
                        tactic_counts[ttype] = {"available": 0, "found": 0, "missed": 0}
                    tactic_counts[ttype]["available"] += 1

                for t in player_found:
                    tactic_counts[t.tactic_type]["found"] += 1
                for t in player_missed:
                    tactic_counts[t.tactic_type]["missed"] += 1
        else:
            prev_tactic_keys = set()

        board.push(move)

    results["summary"] = tactic_counts
    total_available = sum(v["available"] for v in tactic_counts.values())
    total_found = sum(v["found"] for v in tactic_counts.values())
    results["total_tactics_available"] = total_available
    results["total_tactics_found"] = total_found
    results["score_line"] = f"You found {total_found} of {total_available} tactics in this game!"

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Quick test with sample positions (no engine — raw pattern detection)
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = chess.Board(test_fen)

    print(f"Testing position: {test_fen}")
    print(f"Scholar's mate position (Qh5 threatening f7)")
    print()

    tactics = detect_all_tactics(board, chess.WHITE)

    if tactics:
        print(f"Found {len(tactics)} pattern candidates:")
        for t in tactics:
            print(f"  {t}")
    else:
        print("No patterns detected.")

    print()

    # Test discovered attack
    disc_fen = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5"
    board2 = chess.Board(disc_fen)
    print(f"Testing discovered attacks...")
    tactics2 = detect_all_tactics(board2, chess.WHITE)
    for t in tactics2:
        print(f"  {t}")

    print("\nDone. Pattern detectors return candidates — use engine_validate_tactics() to confirm.")
