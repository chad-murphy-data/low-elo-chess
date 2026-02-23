"""
Game state tracking for Stonefish bot.

Maintains board position, move history, piece recency, eval state,
and all context needed for the bot's decision pipeline.

Reuses utility functions from src/features.py for consistency with
the research data pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess

from src.features import (
    PIECE_NAMES,
    PIECE_VALUES,
    material_balance,
    piece_count,
    detect_patterns,
)


# ---------------------------------------------------------------------------
# Per-move record
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    """Record of a single move played in the game."""
    move: chess.Move
    piece_type: Optional[int]       # chess.PAWN, chess.KNIGHT, etc.
    piece_name: str                  # "pawn", "knight", etc.
    is_capture: bool
    is_check: bool
    from_square: int
    to_square: int
    eval_before: Optional[int]      # centipawns, White perspective
    eval_after: Optional[int]       # centipawns, White perspective
    ply: int                        # 1-indexed ply number
    moving_color: chess.Color       # who made this move


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """Tracks the full state of an in-progress game.

    This object is shared between the bot, logger, and CLI.
    All evaluation scores are in centipawns from White's perspective.
    """

    def __init__(
        self,
        bot_color: chess.Color,
        board: Optional[chess.Board] = None,
    ):
        """Initialize game state.

        Args:
            bot_color: chess.WHITE or chess.BLACK -- which side the bot plays.
            board: Starting position. Defaults to standard starting position.
        """
        self.board: chess.Board = board if board is not None else chess.Board()
        self.bot_color: chess.Color = bot_color
        self.move_history: List[MoveRecord] = []
        self.ply: int = 0  # total half-moves played

        # Piece recency tracking: square -> ply when piece there last moved
        # (same pattern as features.py lines 345-346, 464-466)
        self._piece_last_moved: Dict[int, int] = {}

        # Per-player move history for pattern detection
        # (same structure as features.py lines 352, 456-457)
        self._player_move_history: Dict[chess.Color, List[Tuple]] = {
            chess.WHITE: [],
            chess.BLACK: [],
        }

        # Current eval (updated after each move by the engine)
        self._current_eval_cp: Optional[int] = None

        # Plan state placeholder (Phase 3)
        self.current_plan: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def move_number(self) -> int:
        """Current full move number (1-indexed, like chess notation)."""
        return self.board.fullmove_number

    @property
    def is_bot_turn(self) -> bool:
        """True if it's the bot's turn to move."""
        return self.board.turn == self.bot_color

    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.board.is_game_over()

    @property
    def game_result(self) -> Optional[str]:
        """Return game result string or None if game is ongoing."""
        if not self.board.is_game_over():
            return None
        return self.board.result()

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def push_move(
        self,
        move: chess.Move,
        eval_before: Optional[int] = None,
        eval_after: Optional[int] = None,
    ) -> MoveRecord:
        """Execute a move and update all tracking state.

        Args:
            move: The move to play (must be legal).
            eval_before: Position eval before the move (cp, White perspective).
            eval_after: Position eval after the move (cp, White perspective).

        Returns:
            MoveRecord for the move just played.

        Raises:
            ValueError: If the move is not legal.
        """
        if move not in self.board.legal_moves:
            raise ValueError(
                f"Illegal move: {move} in position {self.board.fen()}"
            )

        moving_color = self.board.turn
        moved_piece = self.board.piece_at(move.from_square)
        piece_type = moved_piece.piece_type if moved_piece else None
        piece_name = (
            PIECE_NAMES.get(piece_type, "unknown") if piece_type else "unknown"
        )
        is_capture = self.board.is_capture(move)

        # Update player move history for pattern detection
        self._player_move_history[moving_color].append(
            (move, piece_type)
        )

        # Execute the move
        self.board.push(move)
        self.ply += 1

        # Check if the move gave check
        is_check = self.board.is_check()

        # Update piece recency tracking
        self._piece_last_moved[move.to_square] = self.ply
        if move.from_square in self._piece_last_moved:
            del self._piece_last_moved[move.from_square]

        # Update eval
        if eval_after is not None:
            self._current_eval_cp = eval_after

        # Build record
        record = MoveRecord(
            move=move,
            piece_type=piece_type,
            piece_name=piece_name,
            is_capture=is_capture,
            is_check=is_check,
            from_square=move.from_square,
            to_square=move.to_square,
            eval_before=eval_before,
            eval_after=eval_after,
            ply=self.ply,
            moving_color=moving_color,
        )
        self.move_history.append(record)
        return record

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_opponent_last_move(self) -> Optional[MoveRecord]:
        """Get the last move played by the opponent (relative to bot)."""
        opponent_color = not self.bot_color
        for record in reversed(self.move_history):
            if record.moving_color == opponent_color:
                return record
        return None

    def get_bot_last_move(self) -> Optional[MoveRecord]:
        """Get the last move played by the bot."""
        for record in reversed(self.move_history):
            if record.moving_color == self.bot_color:
                return record
        return None

    def get_last_piece_moved(
        self, color: Optional[chess.Color] = None
    ) -> Optional[int]:
        """Get the piece type of the last piece moved by the given color.

        Args:
            color: Which player's last piece to query. Defaults to bot_color.

        Returns:
            chess.PAWN, chess.KNIGHT, etc., or None if no moves yet.
        """
        if color is None:
            color = self.bot_color
        for record in reversed(self.move_history):
            if record.moving_color == color:
                return record.piece_type
        return None

    def get_last_move_square(
        self, color: Optional[chess.Color] = None
    ) -> Optional[int]:
        """Get the destination square of the last move by the given color.

        Args:
            color: Which player. Defaults to bot_color.

        Returns:
            Square index, or None if no moves yet.
        """
        if color is None:
            color = self.bot_color
        for record in reversed(self.move_history):
            if record.moving_color == color:
                return record.to_square
        return None

    def get_piece_recency(self, square: int) -> Optional[int]:
        """How many plies ago did the piece on `square` last move?

        Returns:
            Number of plies since last move, or None if the piece
            has never moved in this game.
        """
        if square in self._piece_last_moved:
            return self.ply - self._piece_last_moved[square]
        return None

    def get_bot_move_count(self) -> int:
        """How many moves the bot has played so far."""
        return len(self._player_move_history[self.bot_color])

    # ------------------------------------------------------------------
    # Eval queries
    # ------------------------------------------------------------------

    def get_eval(self) -> Optional[int]:
        """Current position evaluation in centipawns (White's perspective)."""
        return self._current_eval_cp

    def set_eval(self, eval_cp: int) -> None:
        """Update the current eval (called after Stockfish evaluation)."""
        self._current_eval_cp = eval_cp

    def get_eval_from_bot_perspective(self) -> Optional[int]:
        """Current eval from the bot's perspective (positive = bot better)."""
        if self._current_eval_cp is None:
            return None
        if self.bot_color == chess.WHITE:
            return self._current_eval_cp
        else:
            return -self._current_eval_cp

    def get_eval_context(self) -> str:
        """Classify current position: 'winning', 'losing', or 'equal'.

        Uses ±100cp threshold (1 pawn) for the classification.
        """
        bot_eval = self.get_eval_from_bot_perspective()
        if bot_eval is None:
            return "equal"  # default when no eval available
        if bot_eval > 100:
            return "winning"
        elif bot_eval < -100:
            return "losing"
        else:
            return "equal"

    # ------------------------------------------------------------------
    # Board state queries
    # ------------------------------------------------------------------

    def get_material_balance(self) -> int:
        """Material balance from the bot's perspective."""
        return material_balance(self.board, self.bot_color)

    def get_piece_count(self) -> int:
        """Total pieces on the board."""
        return piece_count(self.board)

    def detect_pattern(self) -> Tuple[bool, Optional[str]]:
        """Detect if the bot is executing a known attack pattern.

        Returns:
            (is_executing_pattern, pattern_type_or_none)
        """
        if not self.move_history:
            return False, None
        last = self.move_history[-1]
        moved_piece = self.board.piece_at(last.to_square)
        return detect_patterns(
            self.bot_color,
            self._player_move_history,
            last.move,
            moved_piece,
            self.board,
            last.is_capture,
        )
