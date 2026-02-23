"""
Stonefish bot: behaviorally-authentic low-ELO chess engine.

Move selection pipeline (from build spec Section 4):
    Step 1: Did opponent just blunder? Roll miss probability.
    Step 2: Two-gate filter — react mode vs solitaire mode.
    Step 3: Hazard roll — should we blunder this move?
    Step 4: Normal move selection with personality overlays.
    Step 5-6: Blunder candidate generation + attention filter + purpose check.
    Step 7: Execute and log.

All phases implemented.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess

from src.engine import MaiaEngine, MaiaMove, StockfishEngine
from src.features import PIECE_VALUES, PIECE_NAMES, chebyshev_distance
from src.game_state import GameState


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class StonefishBot:
    """The Stonefish low-ELO chess bot.

    Initialized with a target ELO (500, 700, or 900) and loads the
    corresponding parameter file. Manages Maia and Stockfish engines.
    """

    def __init__(
        self,
        elo_target: int,
        maia_engine: Optional[MaiaEngine] = None,
        stockfish_engine: Optional[StockfishEngine] = None,
        config_path: Optional[str] = None,
    ):
        self.elo_target = elo_target
        self.params = self._load_params(config_path)

        self._maia = maia_engine or MaiaEngine()
        self._stockfish = stockfish_engine or StockfishEngine()
        self._engines_started = False

        # Track blunder carry-forward: if no valid blunder candidate
        # was found on the previous hazard fire, try again next move
        self._blunder_carry_forward = False

    def _load_params(self, config_path: Optional[str] = None) -> Dict:
        """Load bot parameters from JSON config file."""
        if config_path is None:
            config_path = CONFIG_DIR / f"bot_{self.elo_target}.json"
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config not found: {path}. "
                f"Expected config/bot_{self.elo_target}.json"
            )
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def start_engines(self) -> None:
        if not self._engines_started:
            self._maia.start()
            self._stockfish.start()
            self._engines_started = True

    def stop_engines(self) -> None:
        if self._engines_started:
            self._maia.stop()
            self._stockfish.stop()
            self._engines_started = False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_position(self, game_state: GameState) -> Optional[int]:
        """Get Stockfish evaluation and update game state."""
        eval_cp = self._stockfish.evaluate(game_state.board)
        if eval_cp is not None:
            game_state.set_eval(eval_cp)
        return eval_cp

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def select_move(self, game_state: GameState) -> Tuple[chess.Move, Dict]:
        """Select a move for the current position.

        Implements the full move selection pipeline from the build spec.

        Returns:
            (selected_move, metadata_dict)
        """
        board = game_state.board
        bot_color = game_state.bot_color

        # --- Get Maia candidates ---
        maia_moves = self._maia.get_top_n_moves(board, n=5)

        if not maia_moves:
            move = random.choice(list(board.legal_moves))
            return move, {
                "mechanism": "random_fallback",
                "maia_rank": None,
                "maia_top5": [],
                "notes": "No Maia candidates, random fallback",
                "mode": "fallback",
            }

        maia_top5_data = [
            {"move": m.move.uci(), "rank": m.rank, "score_cp": m.score_cp}
            for m in maia_moves
        ]

        # ============================================================
        # Step 1: Did opponent just blunder?
        # ============================================================
        opponent_blundered = False
        noticed_blunder = False
        opp_last = game_state.get_opponent_last_move()
        if opp_last is not None and opp_last.eval_before is not None and opp_last.eval_after is not None:
            if opp_last.moving_color == chess.WHITE:
                opp_cp_loss = max(0, opp_last.eval_before - opp_last.eval_after)
            else:
                opp_cp_loss = max(0, opp_last.eval_after - opp_last.eval_before)

            if opp_cp_loss > 100:
                opponent_blundered = True
                miss_rate = self.params["miss_opponent_blunder_rate"]
                if random.random() > miss_rate:
                    noticed_blunder = True

        # ============================================================
        # Step 2: Two-gate filter
        # ============================================================
        gate_result = self._check_gates(game_state)
        mode = gate_result["mode"]

        if mode == "react":
            # React mode: pick from Maia top 2
            react_candidates = maia_moves[:min(2, len(maia_moves))]
            selected_move, selected_rank = self._sample_uniform(react_candidates)
            return selected_move, {
                "mechanism": "react",
                "maia_rank": selected_rank,
                "maia_top5": maia_top5_data,
                "mode": mode,
                "gate_detail": gate_result["detail"],
                "opponent_blundered": opponent_blundered,
                "noticed_blunder": noticed_blunder,
                "notes": (
                    f"React mode ({gate_result['detail']}). "
                    f"Picked rank {selected_rank} from top 2."
                ),
            }

        # ============================================================
        # Step 3: Hazard roll — should this move be a blunder?
        # ============================================================
        bot_move_num = game_state.get_bot_move_count() + 1
        should_blunder = self._hazard_roll(game_state, bot_move_num)

        if should_blunder or self._blunder_carry_forward:
            # ========================================================
            # Steps 5-6: Blunder candidate generation
            # ========================================================
            blunder_result = self._generate_blunder(
                game_state, maia_moves, maia_top5_data
            )
            if blunder_result is not None:
                self._blunder_carry_forward = False
                return blunder_result

            # No valid blunder candidate found — carry forward
            self._blunder_carry_forward = True
            # Fall through to normal move selection

        # ============================================================
        # Step 4: Normal move selection with personality overlays
        # ============================================================
        weighted_moves = self._apply_personality(maia_moves, game_state)
        selected_move, selected_rank = self._weighted_sample(weighted_moves)

        return selected_move, {
            "mechanism": "maia_personality",
            "maia_rank": selected_rank,
            "maia_top5": maia_top5_data,
            "mode": mode,
            "gate_detail": gate_result.get("detail", "none"),
            "opponent_blundered": opponent_blundered,
            "noticed_blunder": noticed_blunder,
            "notes": (
                f"Solitaire mode. Personality-weighted rank {selected_rank}."
            ),
        }

    # ------------------------------------------------------------------
    # Phase 2: Two-gate filter
    # ------------------------------------------------------------------

    def _check_gates(self, game_state: GameState) -> Dict:
        """Run the two-gate opponent model filter.

        Gate 1 — Immediate threat:
            Check, capture, adjacent to major piece, attack on undefended piece.
        Gate 2 — Plan interference:
            Opponent attacks our last-moved piece.

        Returns:
            {"mode": "react" or "solitaire", "detail": str}
        """
        board = game_state.board
        bot_color = game_state.bot_color
        opp_last = game_state.get_opponent_last_move()

        if opp_last is None:
            return {"mode": "solitaire", "detail": "no_opponent_move"}

        # Gate 1: Immediate threats
        if board.is_check():
            return {"mode": "react", "detail": "in_check"}

        if opp_last.is_capture:
            return {"mode": "react", "detail": "piece_captured"}

        opp_landing = opp_last.to_square
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != bot_color:
                continue
            if piece.piece_type in (chess.ROOK, chess.QUEEN, chess.KING):
                if chebyshev_distance(opp_landing, sq) == 1:
                    return {"mode": "react", "detail": f"adjacent_threat_{chess.square_name(sq)}"}

        opp_piece = board.piece_at(opp_landing)
        if opp_piece is not None:
            for sq in board.attacks(opp_landing):
                our_piece = board.piece_at(sq)
                if our_piece is None or our_piece.color != bot_color:
                    continue
                if our_piece.piece_type == chess.KING:
                    continue
                val = PIECE_VALUES.get(our_piece.piece_type, 0)
                if val >= 3 and not board.attackers(bot_color, sq):
                    return {"mode": "react", "detail": f"undefended_{chess.square_name(sq)}_attacked"}

        # Gate 2: Plan interference
        bot_last = game_state.get_bot_last_move()
        if bot_last is not None and opp_piece is not None:
            if bot_last.to_square in board.attacks(opp_landing):
                our_moved_piece = board.piece_at(bot_last.to_square)
                if our_moved_piece and our_moved_piece.color == bot_color:
                    return {"mode": "react", "detail": "plan_interference"}

        return {"mode": "solitaire", "detail": "none"}

    # ------------------------------------------------------------------
    # Phase 3: Personality overlays
    # ------------------------------------------------------------------

    def _apply_personality(
        self,
        candidates: List[MaiaMove],
        game_state: GameState,
    ) -> List[Tuple[MaiaMove, float]]:
        """Apply personality overlays to Maia candidates.

        Overlays: base sampling weights, forward bias, check impulse,
        Scholar's mate logic, capture attraction.
        """
        board = game_state.board
        bot_color = game_state.bot_color
        move_num = game_state.get_bot_move_count() + 1

        sampling_weights = self.params["maia_sampling"]["weights"]
        n = len(candidates)
        base_weights = sampling_weights[:n]
        total = sum(base_weights)
        if total <= 0:
            base_weights = [1.0 / n] * n
        else:
            base_weights = [w / total for w in base_weights]

        results = []
        for i, candidate in enumerate(candidates):
            weight = base_weights[i]
            move = candidate.move

            # Forward bias
            opp_king_sq = board.king(not bot_color)
            if opp_king_sq is not None:
                dist_before = chebyshev_distance(move.from_square, opp_king_sq)
                dist_after = chebyshev_distance(move.to_square, opp_king_sq)
                if dist_after < dist_before:
                    weight *= 1.15
                elif dist_after > dist_before + 1:
                    weight *= 0.90

            # Check impulse
            check_impulse = self.params.get("check_impulse", "medium")
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            if gives_check:
                multiplier = {"high": 2.0, "medium_high": 1.7, "medium": 1.4}.get(
                    check_impulse, 1.2
                )
                weight *= multiplier

            # Scholar's mate logic (first 3 moves as White)
            if (bot_color == chess.WHITE and move_num <= 3
                    and random.random() < self.params.get("scholars_mate_rate", 0.0)):
                target_sq = {1: chess.E4, 2: chess.C4, 3: chess.H5}.get(move_num)
                if target_sq is not None and move.to_square == target_sq:
                    weight *= 5.0

            # Capture attraction
            if board.is_capture(move):
                weight *= 1.2

            results.append((candidate, weight))

        return results

    # ------------------------------------------------------------------
    # Phase 5: Hazard roll
    # ------------------------------------------------------------------

    def _hazard_roll(self, game_state: GameState, bot_move_num: int) -> bool:
        """Determine if this move should be a blunder.

        Uses the hazard curve shape from the spec:
        - Blunder probability ramps steeply through first 20 moves
        - Plateaus after move 20-25
        - Scaled by eval context multiplier (winning/equal/losing)

        Returns:
            True if the hazard fires (bot should blunder this move).
        """
        threshold_move = self.params["first_blunder_threshold_move"]

        # Before the threshold, blunders are very unlikely
        if bot_move_num < threshold_move:
            return False

        # Compute base hazard probability using a sigmoid-like ramp
        plateau = self.params["equal_pos_blunder_rate"]

        if bot_move_num <= 25:
            # Ramp from 0 to plateau over moves threshold..25
            ramp_progress = (bot_move_num - threshold_move) / (25 - threshold_move)
            ramp_progress = max(0.0, min(1.0, ramp_progress))
            base_prob = plateau * ramp_progress
        else:
            # Plateau
            base_prob = plateau

        # Apply eval context multiplier
        eval_context = game_state.get_eval_context()
        if eval_context == "winning":
            base_prob *= self.params.get("winning_multiplier", 1.0)
        elif eval_context == "losing":
            base_prob *= self.params.get("losing_multiplier", 1.0)
        # "equal" uses base_prob as-is

        # Cap at hazard plateau
        hazard_plateau = self.params["hazard_plateau"]
        base_prob = min(base_prob, hazard_plateau)

        return random.random() < base_prob

    # ------------------------------------------------------------------
    # Phase 5: Blunder candidate generation
    # ------------------------------------------------------------------

    def _generate_blunder(
        self,
        game_state: GameState,
        maia_top5: List[MaiaMove],
        maia_top5_data: List[Dict],
    ) -> Optional[Tuple[chess.Move, Dict]]:
        """Generate and validate a blunder candidate.

        Steps:
        1. Roll blunder magnitude (low/mid/high)
        2. Build candidate pool from extended Maia ranks
        3. Apply attention filter (hard exclusions + decay)
        4. Apply purpose check
        5. Return the best valid candidate, or None

        Returns:
            (move, metadata) or None if no valid blunder found.
        """
        board = game_state.board
        bot_color = game_state.bot_color

        # --- Magnitude roll ---
        mag_weights = self.params.get("blunder_magnitude_weights", {})
        mag_roll = random.random()
        if mag_roll < mag_weights.get("low", 0.45):
            magnitude = "low"
        elif mag_roll < mag_weights.get("low", 0.45) + mag_weights.get("mid", 0.35):
            magnitude = "mid"
        else:
            magnitude = "high"

        # --- Build candidate pool ---
        candidates = self._build_blunder_pool(
            game_state, maia_top5, magnitude
        )

        if not candidates:
            return None

        # --- Filter through attention model + purpose check ---
        valid_candidates = []
        for move, source in candidates:
            if not self._passes_hard_exclusions(move, game_state):
                continue
            if not self._passes_purpose_check(move, game_state):
                continue

            # Compute attention score (how likely the bot is to miss
            # the danger of this move)
            attention_score = self._compute_attention_score(move, game_state)

            # For high magnitude, apply repeat-piece bias
            repeat_bias = 1.0
            if magnitude == "high":
                bot_last = game_state.get_bot_last_move()
                if bot_last is not None:
                    moved_piece = board.piece_at(move.from_square)
                    last_piece_sq = bot_last.to_square
                    if move.from_square == last_piece_sq:
                        # Same piece as last turn
                        repeat_bias = self.params.get("repeat_piece_bias", 0.5) * 3.0

            score = attention_score * repeat_bias
            valid_candidates.append((move, source, score))

        if not valid_candidates:
            return None

        # Pick the candidate with the highest combined score
        # (most likely to be "missable" by the bot)
        valid_candidates.sort(key=lambda x: x[2], reverse=True)
        chosen_move, chosen_source, _ = valid_candidates[0]

        # Find the Maia rank if the move was in the top 5
        maia_rank = None
        for m in maia_top5:
            if m.move == chosen_move:
                maia_rank = m.rank
                break

        return chosen_move, {
            "mechanism": f"blunder_{magnitude}",
            "maia_rank": maia_rank,
            "maia_top5": maia_top5_data,
            "mode": "solitaire",
            "gate_detail": "none",
            "blunder_magnitude": magnitude,
            "blunder_source": chosen_source,
            "notes": (
                f"Blunder ({magnitude} magnitude) via {chosen_source}. "
                f"{'Repeat piece.' if chosen_source == 'repeat_piece' else ''}"
            ),
        }

    def _build_blunder_pool(
        self,
        game_state: GameState,
        maia_top5: List[MaiaMove],
        magnitude: str,
    ) -> List[Tuple[chess.Move, str]]:
        """Build the blunder candidate pool.

        Sources (from spec Section 4, Step 6):
        - Maia ranks 6-15 (extended candidates)
        - Maia top 5 moves that lose material (good-looking but bad)
        - Checks and captures that lose material
        - Same piece as last turn with a target

        Returns:
            List of (move, source_label) tuples.
        """
        board = game_state.board
        bot_color = game_state.bot_color
        candidates = []

        # Source 1: Extended Maia candidates (ranks beyond top 5)
        # For low magnitude: ranks 6-10
        # For mid magnitude: ranks 8-15
        # For high magnitude: all sources
        if magnitude == "low":
            extended = self._maia.get_top_n_moves(board, n=10)
            for m in extended[5:]:  # ranks 6-10
                candidates.append((m.move, "maia_extended"))
        elif magnitude == "mid":
            extended = self._maia.get_top_n_moves(board, n=15)
            for m in extended[7:]:  # ranks 8-15
                candidates.append((m.move, "maia_extended"))
        else:
            extended = self._maia.get_top_n_moves(board, n=15)
            for m in extended[5:]:  # ranks 6-15
                candidates.append((m.move, "maia_extended"))

        # Source 2: Good-looking checks/captures that lose material
        for move in board.legal_moves:
            board.push(move)
            gives_check = board.is_check()
            board.pop()

            is_capture = board.is_capture(move)
            if not gives_check and not is_capture:
                continue

            # Check if this move is already in top 5 (skip if so)
            if any(m.move == move for m in maia_top5):
                continue

            # Is this move actually bad? Quick check: is the landing
            # square defended by more pieces than we have attacking?
            to_sq = move.to_square
            attackers = len(board.attackers(bot_color, to_sq))
            defenders = len(board.attackers(not bot_color, to_sq))
            if defenders > attackers:
                candidates.append((move, "tactical_trap"))

        # Source 3: Same piece as last turn (repeat piece)
        if magnitude in ("mid", "high"):
            bot_last = game_state.get_bot_last_move()
            if bot_last is not None:
                last_piece_sq = bot_last.to_square
                piece = board.piece_at(last_piece_sq)
                if piece and piece.color == bot_color:
                    for move in board.legal_moves:
                        if move.from_square == last_piece_sq:
                            if any(m.move == move for m in maia_top5):
                                continue
                            candidates.append((move, "repeat_piece"))

        # Deduplicate by move
        seen = set()
        deduped = []
        for move, source in candidates:
            if move not in seen:
                seen.add(move)
                deduped.append((move, source))

        return deduped

    # ------------------------------------------------------------------
    # Phase 4: Attention filter
    # ------------------------------------------------------------------

    def _passes_hard_exclusions(
        self, move: chess.Move, game_state: GameState
    ) -> bool:
        """Check hard exclusion rules (spec Section 2.1).

        A blunder candidate is EXCLUDED (cannot be played) if:
        1. Landing square was covered by opponent's last move
        2. Nearest capturing piece is exactly 1 square away
        3. Move reverses the piece to its previous square

        Returns:
            True if the move passes (is NOT excluded).
        """
        board = game_state.board
        bot_color = game_state.bot_color
        opp_last = game_state.get_opponent_last_move()

        # Exclusion 1: Landing square covered by opponent's last move
        if opp_last is not None:
            opp_piece = board.piece_at(opp_last.to_square)
            if opp_piece is not None:
                opp_attacks = board.attacks(opp_last.to_square)
                if move.to_square in opp_attacks:
                    return False  # Can't miss something traceable to last move

        # Exclusion 2: Nearest capturing piece is 1 square away
        for sq in board.attackers(not bot_color, move.to_square):
            if chebyshev_distance(sq, move.to_square) == 1:
                return False  # Can't miss a capturer right next to you

        # Exclusion 3: Move reverses piece to previous square
        bot_last = game_state.get_bot_last_move()
        if bot_last is not None:
            if (move.from_square == bot_last.to_square
                    and move.to_square == bot_last.from_square):
                return False  # Can't reverse the exact same piece

        return True

    def _passes_purpose_check(
        self, move: chess.Move, game_state: GameState
    ) -> bool:
        """Check if a blunder candidate has an articulable purpose.

        From spec Section 4.2: every blunder must pass
        "I played that move because..."

        Valid purposes:
        - Attacking an opponent piece
        - Giving check or threatening checkmate
        - Continuing a capture sequence
        - Escaping an attack on the current piece
        - Continuing same piece trajectory (with a target)

        Returns:
            True if the move has a valid purpose.
        """
        board = game_state.board
        bot_color = game_state.bot_color
        moved_piece = board.piece_at(move.from_square)

        # Purpose 1: Attacking an opponent piece
        board.push(move)
        attacks_after = board.attacks(move.to_square)
        board.pop()
        for sq in attacks_after:
            target = board.piece_at(sq)
            if target is not None and target.color != bot_color:
                return True  # Attacking an opponent piece

        # Purpose 2: Giving check
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            return True

        # Purpose 3: Is a capture (continuing a capture sequence)
        if board.is_capture(move):
            return True

        # Purpose 4: Escaping an attack
        if moved_piece is not None:
            if board.is_attacked_by(not bot_color, move.from_square):
                return True  # Piece was under attack, moving it

        # Purpose 5: Continuing same piece from last turn (with a target)
        bot_last = game_state.get_bot_last_move()
        if bot_last is not None and move.from_square == bot_last.to_square:
            # Same piece — check if it has a target on the new square
            board.push(move)
            new_attacks = board.attacks(move.to_square)
            board.pop()
            for sq in new_attacks:
                target = board.piece_at(sq)
                if target is not None and target.color != bot_color:
                    return True

        # Special: queen moves with no target and no escape = invalid
        if moved_piece is not None and moved_piece.piece_type == chess.QUEEN:
            return False  # Queen relocating with no purpose

        # For non-queen pieces, allow pawn pushes and development
        if moved_piece is not None and moved_piece.piece_type == chess.PAWN:
            return True  # Pawns always have purpose (advancing)

        # Knight/bishop development (moving forward in opening)
        if moved_piece is not None and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            move_num = game_state.get_bot_move_count() + 1
            if move_num <= 10:
                return True  # Development moves are valid purpose

        return False

    def _compute_attention_score(
        self, move: chess.Move, game_state: GameState
    ) -> float:
        """Compute attention score for a blunder candidate.

        Models how "missable" a threat to this move is, using two
        independent dimensions from spec Section 2.2:

            attention = f(distance) * g(recency)

        Higher score = more missable (bot more likely to play this blunder).

        f(distance): decays toward 0 as capturer distance increases beyond 2
        g(recency): decays toward 0 over 3-5 moves since threat became relevant

        Returns:
            Float between 0.0 (unmissable) and 1.0 (easily missed).
        """
        board = game_state.board
        bot_color = game_state.bot_color
        attention_window = self.params["attention_decay_window"]

        # Find nearest opponent attacker of the landing square
        attackers = board.attackers(not bot_color, move.to_square)
        if not attackers:
            return 1.0  # No attackers → fully missable (actually safe)

        # Distance dimension: f(distance)
        min_dist = 99
        nearest_attacker_sq = None
        for sq in attackers:
            d = chebyshev_distance(sq, move.to_square)
            if d < min_dist:
                min_dist = d
                nearest_attacker_sq = sq

        if min_dist <= 1:
            return 0.0  # Hard exclusion should have caught this, but safety
        elif min_dist == 2:
            f_dist = 0.3  # Close but not adjacent — partially visible
        elif min_dist == 3:
            f_dist = 0.6  # Moderate distance
        elif min_dist == 4:
            f_dist = 0.8  # Far
        else:
            f_dist = 0.95  # Very far — almost invisible

        # Recency dimension: g(recency)
        # How many plies ago did the nearest attacker last move?
        recency = game_state.get_piece_recency(nearest_attacker_sq)
        if recency is None:
            recency = 100  # Piece hasn't moved → very old

        # Convert ply recency to "moves ago" (roughly halve for player moves)
        moves_ago = recency / 2.0

        if moves_ago <= 0.5:
            g_rec = 0.0  # Just moved — opponent's last move, can't miss
        elif moves_ago <= 1.0:
            g_rec = 0.2  # One move ago — hard to miss
        elif moves_ago < attention_window:
            # Linear decay within attention window
            g_rec = (moves_ago - 1.0) / (attention_window - 1.0)
            g_rec = min(1.0, max(0.0, g_rec))
        else:
            g_rec = 1.0  # Outside attention window — fully forgotten

        return f_dist * g_rec

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _maia_weighted_sample(
        self, candidates: List[MaiaMove]
    ) -> Tuple[chess.Move, int]:
        """Sample from Maia candidates using ELO-specific weights."""
        sampling_weights = self.params["maia_sampling"]["weights"]
        n = len(candidates)
        weights = sampling_weights[:n]
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in weights]
        chosen_idx = random.choices(range(n), weights=weights, k=1)[0]
        return candidates[chosen_idx].move, candidates[chosen_idx].rank

    def _sample_uniform(
        self, candidates: List[MaiaMove]
    ) -> Tuple[chess.Move, int]:
        """Sample uniformly (for react mode)."""
        chosen = random.choice(candidates)
        return chosen.move, chosen.rank

    def _weighted_sample(
        self, weighted_candidates: List[Tuple[MaiaMove, float]]
    ) -> Tuple[chess.Move, int]:
        """Sample from personality-weighted candidates."""
        if not weighted_candidates:
            raise ValueError("No candidates to sample from")
        moves = [wc[0] for wc in weighted_candidates]
        weights = [max(0.001, wc[1]) for wc in weighted_candidates]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen_idx = random.choices(range(len(moves)), weights=weights, k=1)[0]
        return moves[chosen_idx].move, moves[chosen_idx].rank

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start_engines()
        return self

    def __exit__(self, *exc):
        self.stop_engines()
