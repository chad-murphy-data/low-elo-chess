"""
Game logging to SQLite for Stonefish bot.

Logs every game played and every individual move with full context,
enabling post-hoc analysis of bot behavior and calibration.

Database: data/stonefish_games.db
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "stonefish_games.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_GAMES_TABLE = """
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    bot_elo     INTEGER NOT NULL,
    bot_color   TEXT NOT NULL,
    opponent_type TEXT NOT NULL DEFAULT 'human',
    time_control TEXT,
    result      TEXT,
    total_moves INTEGER,
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);
"""

CREATE_MOVES_TABLE = """
CREATE TABLE IF NOT EXISTS moves (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    ply         INTEGER NOT NULL,
    player      TEXT NOT NULL,
    fen_before  TEXT NOT NULL,
    move_uci    TEXT NOT NULL,
    move_san    TEXT,
    piece_moved TEXT,
    is_capture  INTEGER,
    is_check    INTEGER,
    eval_before INTEGER,
    eval_after  INTEGER,
    cp_loss     INTEGER,
    was_blunder INTEGER DEFAULT 0,
    mechanism   TEXT,
    maia_rank   INTEGER,
    maia_top5   TEXT,
    notes       TEXT,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

CREATE_MOVES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id);
"""


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------

class GameLogger:
    """Logs games and moves to a SQLite database.

    Usage:
        with GameLogger() as logger:
            gid = logger.start_game(500, "white")
            logger.log_move(gid, ...)
            logger.end_game(gid, "1-0")
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        """Open database connection and ensure tables exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(CREATE_GAMES_TABLE)
        self._conn.execute(CREATE_MOVES_TABLE)
        self._conn.execute(CREATE_MOVES_INDEX)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def start_game(
        self,
        bot_elo: int,
        bot_color: str,
        opponent_type: str = "human",
        time_control: Optional[str] = None,
    ) -> str:
        """Start logging a new game.

        Returns:
            game_id (short UUID string) for this game.
        """
        if self._conn is None:
            raise RuntimeError("Logger not opened. Call open() first.")

        game_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        self._conn.execute(
            "INSERT INTO games (game_id, bot_elo, bot_color, opponent_type, "
            "time_control, started_at) VALUES (?, ?, ?, ?, ?, ?)",
            (game_id, bot_elo, bot_color, opponent_type, time_control, now),
        )
        self._conn.commit()
        return game_id

    def log_move(
        self,
        game_id: str,
        move_number: int,
        ply: int,
        player: str,
        fen_before: str,
        move_uci: str,
        move_san: Optional[str] = None,
        piece_moved: Optional[str] = None,
        is_capture: bool = False,
        is_check: bool = False,
        eval_before: Optional[int] = None,
        eval_after: Optional[int] = None,
        cp_loss: Optional[int] = None,
        was_blunder: bool = False,
        mechanism: Optional[str] = None,
        maia_rank: Optional[int] = None,
        maia_top5: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Log a single move to the database.

        Args:
            game_id: Game identifier from start_game().
            move_number: Full move number (1-indexed).
            ply: Half-move count (1-indexed).
            player: "bot" or "human".
            fen_before: FEN string of position before the move.
            move_uci: Move in UCI notation (e.g., "e2e4").
            move_san: Move in SAN notation (e.g., "e4").
            piece_moved: Piece name (e.g., "pawn", "knight").
            is_capture: Whether the move was a capture.
            is_check: Whether the move gave check.
            eval_before: Centipawns before move (White perspective).
            eval_after: Centipawns after move (White perspective).
            cp_loss: Centipawn loss for this move.
            was_blunder: Whether this move is classified as a blunder (CPL > 100).
            mechanism: How the move was selected (e.g., "maia_sample").
            maia_rank: Which Maia rank was selected (1-5).
            maia_top5: JSON string of Maia's top 5 moves.
            notes: Free-form notes.
        """
        if self._conn is None:
            raise RuntimeError("Logger not opened. Call open() first.")

        self._conn.execute(
            "INSERT INTO moves (game_id, move_number, ply, player, fen_before, "
            "move_uci, move_san, piece_moved, is_capture, is_check, "
            "eval_before, eval_after, cp_loss, was_blunder, mechanism, "
            "maia_rank, maia_top5, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                game_id, move_number, ply, player, fen_before,
                move_uci, move_san, piece_moved,
                int(is_capture), int(is_check),
                eval_before, eval_after, cp_loss, int(was_blunder),
                mechanism, maia_rank, maia_top5, notes,
            ),
        )
        self._conn.commit()

    def end_game(
        self,
        game_id: str,
        result: str,
        total_moves: Optional[int] = None,
    ) -> None:
        """Mark a game as complete.

        Args:
            game_id: Game identifier.
            result: Game result ("1-0", "0-1", "1/2-1/2", or "*" for aborted).
            total_moves: Total number of full moves played.
        """
        if self._conn is None:
            raise RuntimeError("Logger not opened. Call open() first.")

        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE games SET result = ?, total_moves = ?, ended_at = ? "
            "WHERE game_id = ?",
            (result, total_moves, now, game_id),
        )
        self._conn.commit()

    def get_game_count(self) -> int:
        """Return total number of games in the database."""
        if self._conn is None:
            raise RuntimeError("Logger not opened. Call open() first.")
        cursor = self._conn.execute("SELECT COUNT(*) FROM games")
        return cursor.fetchone()[0]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
