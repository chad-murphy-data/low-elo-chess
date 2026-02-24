"""
Engine management for Stonefish: wraps Maia (via lc0) and Stockfish.

Both engines use the python-chess UCI interface. Maia is used for
human-like move generation; Stockfish for position evaluation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chess
import chess.engine


# ---------------------------------------------------------------------------
# Default paths (project-relative, overridable via env vars or constructor)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LC0_PATH = _PROJECT_ROOT / "lc0_dir" / "lc0.exe"
DEFAULT_MAIA_WEIGHTS = _PROJECT_ROOT / "lc0_dir" / "maia-1100.pb.gz"
DEFAULT_SF_PATH = (
    _PROJECT_ROOT
    / "stockfish_dir"
    / "stockfish"
    / "stockfish-windows-x86-64-avx2.exe"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MaiaMove:
    """A single move candidate from Maia with its rank position.

    The rank (1 = top choice) is determined by Maia's policy network.
    At nodes=1, all scores are identical — the *ordering* is the signal.
    """
    move: chess.Move
    rank: int             # 1 = Maia's top choice, 5 = 5th choice
    score_cp: int         # raw score reported by lc0 (often identical across ranks)


# ---------------------------------------------------------------------------
# Maia engine (via lc0)
# ---------------------------------------------------------------------------

class MaiaEngine:
    """Wraps lc0 with Maia weights for human-like move prediction.

    CRITICAL: Uses nodes=1 to get a single neural network evaluation
    with no search tree. This produces human-like move probabilities
    rather than engine-optimal play. The ranking order from the policy
    network is the meaningful signal; scores may be identical.
    """

    def __init__(
        self,
        lc0_path: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        self._lc0_path = str(
            lc0_path or os.environ.get("LC0_PATH", DEFAULT_LC0_PATH)
        )
        self._weights_path = str(
            weights_path or os.environ.get("MAIA_WEIGHTS", DEFAULT_MAIA_WEIGHTS)
        )
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self) -> None:
        """Start the lc0 engine process and configure Maia weights."""
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(self._lc0_path)
        self._engine.configure({"WeightsFile": self._weights_path})

    def stop(self) -> None:
        """Gracefully shut down the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def get_top_n_moves(self, board: chess.Board, n: int = 5) -> List[MaiaMove]:
        """Get Maia's top-N move candidates ranked by policy network.

        The ranking order reflects what a human at ~1100 ELO would most
        likely play. Rank 1 = most likely human move, rank 5 = 5th most
        likely.

        Args:
            board: Current board position.
            n: Number of top moves to return (default 5).

        Returns:
            List of MaiaMove, sorted by rank ascending (best first).
            May return fewer than n if the position has fewer legal moves.
        """
        if self._engine is None:
            raise RuntimeError("MaiaEngine not started. Call start() first.")

        # Clamp n to number of legal moves
        n_legal = board.legal_moves.count()
        n = min(n, n_legal)
        if n == 0:
            return []

        # multipv=n returns n separate info dicts, ranked by policy
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=1),
            multipv=n,
        )

        results = []
        for rank_idx, info in enumerate(infos):
            pv = info.get("pv", [])
            if not pv:
                continue
            move = pv[0]

            # Force all promotions to queen (Maia sometimes picks knight/rook/bishop)
            if move.promotion is not None and move.promotion != chess.QUEEN:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

            # Extract raw score (usually identical across ranks at nodes=1)
            score = info.get("score", None)
            cp = 0
            if score is not None:
                cp = score.white().score(mate_score=10000) or 0

            results.append(MaiaMove(
                move=move,
                rank=rank_idx + 1,
                score_cp=cp,
            ))

        return results

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# Stockfish engine
# ---------------------------------------------------------------------------

class StockfishEngine:
    """Wraps Stockfish for position evaluation.

    Follows the same UCI pattern as src/stockfish_eval.py.
    """

    def __init__(
        self,
        sf_path: Optional[str] = None,
        depth: int = 12,
        threads: Optional[int] = None,
        hash_mb: int = 128,
    ):
        self._sf_path = str(
            sf_path or os.environ.get("SF_PATH", DEFAULT_SF_PATH)
        )
        self._depth = depth
        self._threads = threads or max(1, (os.cpu_count() or 4) - 2)
        self._hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self) -> None:
        """Start the Stockfish engine process."""
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(self._sf_path)
        self._engine.configure({
            "Threads": self._threads,
            "Hash": self._hash_mb,
        })

    def stop(self) -> None:
        """Gracefully shut down the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def evaluate(self, board: chess.Board) -> Optional[int]:
        """Evaluate position, returning centipawns from White's perspective.

        Mate scores are mapped to +/- 10000, consistent with the
        convention in src/stockfish_eval.py and src/features.py.

        Returns:
            Centipawn score (int), or None on error.
        """
        if self._engine is None:
            raise RuntimeError("StockfishEngine not started. Call start() first.")

        info = self._engine.analyse(
            board, chess.engine.Limit(depth=self._depth)
        )
        score = info.get("score")
        if score is None:
            return None

        cp = score.white().score(mate_score=10000)
        return cp

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get Stockfish's best move for the position.

        Returns:
            chess.Move, or None if no move available.
        """
        if self._engine is None:
            raise RuntimeError("StockfishEngine not started. Call start() first.")

        result = self._engine.play(
            board, chess.engine.Limit(depth=self._depth)
        )
        return result.move

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
