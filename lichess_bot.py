"""
Stonefish Lichess Bot
=====================
Connects the Stonefish low-ELO chess bot to Lichess for live games.

Usage:
    # Listen for challenges at ELO 500 (default)
    python lichess_bot.py --elo 500

    # Challenge a specific player
    python lichess_bot.py --elo 500 --challenge USERNAME

    # With custom time control
    python lichess_bot.py --elo 700 --challenge USERNAME --clock 15+10

Setup:
    1. Create a Lichess BOT account (or upgrade an existing one)
    2. Generate an API token at https://lichess.org/account/oauth/token
       - Enable scopes: bot:play, challenge:read, challenge:write
    3. Save the token to lichess.token in this directory
       OR set the LICHESS_TOKEN environment variable
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import random
from queue import Queue
from contextlib import contextmanager
from typing import Optional, Dict

import chess
import chess.engine
import berserk

from src.engine import MaiaEngine, StockfishEngine, DEFAULT_SF_PATH
from src.bot import StonefishBot
from src.game_state import GameState
from src.game_logger import GameLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stonefish-lichess")


# ---------------------------------------------------------------------------
# Stockfish Pool -- manages N engine instances for concurrent games
# ---------------------------------------------------------------------------

class StockfishPool:
    """Thread-safe pool of Stockfish engine instances.

    Each game acquires an engine for the duration of a move computation,
    then returns it. This allows N concurrent games with N engines.
    """

    def __init__(self, path: str, num_engines: int = 3,
                 threads: int = 2, hash_mb: int = 256):
        self._engines: Queue = Queue()
        self._num_engines = num_engines
        for i in range(num_engines):
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": threads, "Hash": hash_mb})
            self._engines.put(engine)
            log.info(f"Stockfish pool engine {i+1}/{num_engines} ready")

    @contextmanager
    def acquire(self):
        """Acquire an engine for use. Blocks if all are busy."""
        engine = self._engines.get()
        try:
            yield engine
        finally:
            self._engines.put(engine)

    def shutdown(self):
        """Stop all engines."""
        while not self._engines.empty():
            try:
                engine = self._engines.get_nowait()
                engine.quit()
            except Exception:
                pass
        log.info("All Stockfish pool engines stopped")


# ---------------------------------------------------------------------------
# Game Handler -- manages a single Lichess game
# ---------------------------------------------------------------------------

class GameHandler:
    """Handles a single Lichess game in its own thread.

    Each game gets its own StonefishBot instance with independent
    game state, Maia engine, and SQLite logger.
    """

    def __init__(
        self,
        client: berserk.Client,
        game_id: str,
        bot_id: str,
        elo_target: int,
        sf_pool: StockfishPool,
        maia_engine: MaiaEngine,
    ):
        self.client = client
        self.game_id = game_id
        self.bot_id = bot_id
        self.elo_target = elo_target
        self.sf_pool = sf_pool
        self.maia_engine = maia_engine

        self.board = chess.Board()
        self.our_color: Optional[chess.Color] = None
        self.opponent_name = "unknown"

        # Stonefish bot + game state + logger are created on gameFull
        self.bot: Optional[StonefishBot] = None
        self.game_state: Optional[GameState] = None
        self.logger = GameLogger()
        self.logger.open()
        self._game_db_id: Optional[str] = None

        # Track which half-moves we've already processed
        self._last_processed_moves = 0

    def run(self):
        """Main game loop -- stream events and respond."""
        log.info(f"Game {self.game_id}: streaming game state...")
        backoff = 1
        while True:
            try:
                for event in self.client.bots.stream_game_state(self.game_id):
                    backoff = 1
                    event_type = event.get("type", "")
                    if event_type == "gameFull":
                        self._handle_game_full(event)
                    elif event_type == "gameState":
                        self._apply_state(event)
                    elif event_type == "chatLine":
                        self._handle_chat(event)
                    elif event_type == "opponentGone":
                        log.info(f"Game {self.game_id}: opponent disconnected")
                break  # Stream ended cleanly
            except berserk.exceptions.ResponseError as e:
                log.error(f"Game {self.game_id}: API error: {e}")
                break
            except Exception as e:
                log.warning(f"Game {self.game_id}: stream error: {e}")
                log.info(f"Game {self.game_id}: reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        # Game over -- finalize
        self._on_game_end()
        log.info(f"Game {self.game_id}: handler finished")

    def _handle_game_full(self, event):
        """Process the initial gameFull event.

        On reconnect, the stream re-sends gameFull. We detect this by
        checking if our_color is already set and skip re-initialization
        to preserve game state.
        """
        white = event.get("white", {})
        black = event.get("black", {})
        white_id = white.get("id", white.get("name", ""))
        black_id = black.get("id", black.get("name", ""))

        is_reconnect = self.our_color is not None

        self.our_color = chess.WHITE if white_id == self.bot_id else chess.BLACK
        self.opponent_name = black_id if self.our_color == chess.WHITE else white_id
        color_str = "white" if self.our_color == chess.WHITE else "black"

        if is_reconnect:
            log.info(f"Game {self.game_id}: reconnected -- we are {color_str} (state preserved)")
        else:
            log.info(f"Game {self.game_id}: {white_id} (W) vs {black_id} (B) -- we are {color_str}")

            # Initialize bot and game state
            self.bot = StonefishBot(
                elo_target=self.elo_target,
                maia_engine=self.maia_engine,
            )
            # Bot uses its own engines internally but we override Stockfish
            # with the pool for each move. Maia is shared (thread-safe for
            # nodes=1 single analysis calls with the GIL).
            self.bot._engines_started = True  # We manage engines externally

            self.game_state = GameState(bot_color=self.our_color)

            # Start game log
            self._game_db_id = self.logger.start_game(
                bot_elo=self.elo_target,
                bot_color=color_str,
                opponent_type=f"lichess:{self.opponent_name}",
            )

            # Send greeting
            self._send_chat(
                f"GL! I'm Stonefish {self.elo_target}. "
                f"I play like a real {self.elo_target}-rated player."
            )

        # Apply initial state (always -- catches up on any moves we missed)
        state = event.get("state", {})
        self._apply_state(state)

    def _apply_state(self, state):
        """Process a game state update -- rebuild board, detect new moves, play if our turn."""
        if self.our_color is None or self.bot is None or self.game_state is None:
            return

        moves_str = state.get("moves", "")

        # Rebuild board from move list
        self.board = chess.Board()
        move_list = []
        if moves_str:
            for uci_str in moves_str.split():
                try:
                    move = self.board.push_uci(uci_str)
                    move_list.append(move)
                except ValueError:
                    log.error(f"Game {self.game_id}: invalid move in stream: {uci_str}")
                    return

        # Check if game is over
        status = state.get("status", "started")
        if status != "started" or self.board.is_game_over():
            return

        current_half_moves = len(move_list)

        # Sync our GameState with the board by replaying any new moves
        if current_half_moves > self._last_processed_moves:
            for i in range(self._last_processed_moves, current_half_moves):
                move = move_list[i]
                move_color = chess.WHITE if (i % 2 == 0) else chess.BLACK

                if move_color != self.our_color:
                    # Opponent's move -- push it into our game state
                    # Build board BEFORE this move for eval
                    temp_board = chess.Board()
                    for j in range(i):
                        temp_board.push(move_list[j])

                    # Get eval before opponent's move
                    eval_before = None
                    try:
                        with self.sf_pool.acquire() as sf_engine:
                            info = sf_engine.analyse(
                                temp_board, chess.engine.Limit(depth=12)
                            )
                            score = info.get("score")
                            if score is not None:
                                eval_before = score.white().score(mate_score=10000)
                    except Exception as e:
                        log.warning(f"Game {self.game_id}: eval error: {e}")

                    # Push move into game state
                    self.game_state.board = temp_board
                    try:
                        record = self.game_state.push_move(move, eval_before=eval_before)
                    except ValueError as e:
                        log.error(f"Game {self.game_id}: push error: {e}")
                        continue

                    # Get eval after
                    eval_after = None
                    try:
                        with self.sf_pool.acquire() as sf_engine:
                            info = sf_engine.analyse(
                                self.game_state.board,
                                chess.engine.Limit(depth=12),
                            )
                            score = info.get("score")
                            if score is not None:
                                eval_after = score.white().score(mate_score=10000)
                                self.game_state.set_eval(eval_after)
                                record.eval_after = eval_after
                    except Exception as e:
                        log.warning(f"Game {self.game_id}: eval after error: {e}")

                    # Log opponent move
                    if self._game_db_id:
                        cp_loss = None
                        if eval_before is not None and eval_after is not None:
                            if move_color == chess.WHITE:
                                cp_loss = max(0, eval_before - eval_after)
                            else:
                                cp_loss = max(0, eval_after - eval_before)

                        self.logger.log_move(
                            game_id=self._game_db_id,
                            move_number=self.game_state.move_number,
                            ply=self.game_state.ply,
                            player="opponent",
                            fen_before=temp_board.fen(),
                            move_uci=move.uci(),
                            move_san=temp_board.san(move),
                            piece_moved=record.piece_name,
                            is_capture=record.is_capture,
                            is_check=record.is_check,
                            eval_before=eval_before,
                            eval_after=eval_after,
                            cp_loss=cp_loss,
                            was_blunder=(cp_loss is not None and cp_loss > 100),
                            mechanism="opponent",
                        )

            self._last_processed_moves = current_half_moves

        # Ensure our game_state board matches the current board
        self.game_state.board = self.board.copy()

        # Is it our turn?
        if self.board.turn != self.our_color:
            return

        # === OUR TURN -- compute and play ===
        move_num = self.board.fullmove_number
        log.info(f"Game {self.game_id}: move {move_num}, thinking...")

        try:
            start = time.time()

            # Get eval before our move
            eval_before = self.game_state.get_eval()
            fen_before = self.board.fen()

            # Temporarily set bot's Stockfish to a pool engine for eval
            with self.sf_pool.acquire() as sf_engine:
                self.bot._stockfish._engine = sf_engine
                try:
                    # Evaluate current position
                    eval_cp = self.bot.evaluate_position(self.game_state)
                    eval_before = eval_cp

                    # Select move
                    move, metadata = self.bot.select_move(self.game_state)
                finally:
                    self.bot._stockfish._engine = None

            elapsed = time.time() - start

            # Push move into our game state
            san = self.game_state.board.san(move)
            record = self.game_state.push_move(move, eval_before=eval_before)

            # Eval after our move
            eval_after = None
            try:
                with self.sf_pool.acquire() as sf_engine:
                    self.bot._stockfish._engine = sf_engine
                    try:
                        eval_after = self.bot.evaluate_position(self.game_state)
                        record.eval_after = eval_after
                    finally:
                        self.bot._stockfish._engine = None
            except Exception as e:
                log.warning(f"Game {self.game_id}: post-move eval error: {e}")

            # Compute cp_loss
            cp_loss = None
            if eval_before is not None and eval_after is not None:
                if self.our_color == chess.WHITE:
                    cp_loss = max(0, eval_before - eval_after)
                else:
                    cp_loss = max(0, eval_after - eval_before)

            was_blunder = cp_loss is not None and cp_loss > 100

            # Log our move
            if self._game_db_id:
                self.logger.log_move(
                    game_id=self._game_db_id,
                    move_number=self.game_state.move_number,
                    ply=self.game_state.ply,
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
                    mechanism=metadata.get("mechanism", "unknown"),
                    maia_rank=metadata.get("maia_rank"),
                    maia_top5=json.dumps(metadata.get("maia_top5", [])),
                    notes=metadata.get("notes"),
                )

            # Log to console
            mechanism = metadata.get("mechanism", "?")
            maia_rank = metadata.get("maia_rank", "?")
            blunder_tag = ""
            if was_blunder and cp_loss is not None:
                blunder_tag = f" [BLUNDER: {cp_loss}cp]"
            log.info(
                f"Game {self.game_id}: playing {san} ({move.uci()}) "
                f"[{mechanism}, rank {maia_rank}, {elapsed:.1f}s]{blunder_tag}"
            )

            # Send the move to Lichess
            try:
                self.client.bots.make_move(self.game_id, move.uci())
            except berserk.exceptions.ResponseError as e:
                log.error(f"Game {self.game_id}: failed to make move {move.uci()}: {e}")

            self._last_processed_moves += 1

        except Exception as e:
            log.error(f"Game {self.game_id}: engine error on move {move_num}: {e}", exc_info=True)
            # Fallback: play a random legal move
            try:
                fallback_move = random.choice(list(self.board.legal_moves))
                self.client.bots.make_move(self.game_id, fallback_move.uci())
                self._last_processed_moves += 1
                log.info(f"Game {self.game_id}: played fallback {fallback_move.uci()}")
            except Exception as e2:
                log.error(f"Game {self.game_id}: fallback also failed: {e2}")

    def _handle_chat(self, event):
        """Process incoming chat messages."""
        username = event.get("username", "")
        text = event.get("text", "")
        if username == self.bot_id:
            return
        log.info(f"Game {self.game_id}: chat from {username}: {text}")

    def _on_game_end(self):
        """Finalize game logging."""
        if self._game_db_id is not None:
            result = self.board.result() if self.board.is_game_over() else "*"
            move_number = self.game_state.move_number if self.game_state else 0
            self.logger.end_game(
                self._game_db_id,
                result=result,
                total_moves=move_number,
            )
            log.info(f"Game {self.game_id}: logged to DB (result: {result})")

        self.logger.close()

    def _send_chat(self, message: str):
        """Send a chat message to the player room."""
        if not message:
            return
        try:
            # Lichess chat has a ~400 character limit
            for chunk in self._split_message(message, max_len=400):
                self.client.bots.post_message(self.game_id, chunk)
        except Exception as e:
            log.warning(f"Game {self.game_id}: chat failed: {e}")

    @staticmethod
    def _split_message(message: str, max_len: int = 400) -> list:
        """Split a message into chunks that fit Lichess chat limits."""
        if len(message) <= max_len:
            return [message]
        lines = message.split("\n")
        chunks = []
        current = ""
        for line in lines:
            if current and len(current) + len(line) + 1 > max_len:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        return chunks


# ---------------------------------------------------------------------------
# Main Bot -- listens for events, manages games
# ---------------------------------------------------------------------------

class StonefishLichessBot:
    """Main bot that listens for Lichess events and manages concurrent games."""

    SUPPORTED_VARIANTS = {"standard", "fromPosition"}
    MIN_TIME_CONTROL = 300  # 5 minutes minimum

    def __init__(
        self,
        token: str,
        elo_target: int = 500,
        max_games: int = 3,
        rated_only: bool = False,
        casual_only: bool = False,
    ):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session)
        self.elo_target = elo_target
        self.max_games = max_games
        self.rated_only = rated_only
        self.casual_only = casual_only

        self.sf_pool: Optional[StockfishPool] = None
        self.maia_engine: Optional[MaiaEngine] = None
        self.active_games: Dict[str, threading.Thread] = {}
        self.bot_id: Optional[str] = None

    def start(self):
        """Start the bot: init engines, connect, listen for events."""
        account = self.client.account.get()
        self.bot_id = account.get("id", "")
        title = account.get("title", "")
        log.info(f"Logged in as: {self.bot_id} (title: {title})")

        if title != "BOT":
            log.warning(
                "Account is NOT a BOT account! "
                "Upgrade at lichess.org/account/bot or via API."
            )

        # Start engines
        self._start_engines()

        try:
            self._event_loop_with_reconnect()
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self._stop_engines()

    def _start_engines(self):
        """Initialize the Stockfish pool and Maia engine."""
        sf_path = str(DEFAULT_SF_PATH)
        log.info(f"Starting Stockfish pool ({self.max_games} engines)...")
        self.sf_pool = StockfishPool(
            sf_path,
            num_engines=self.max_games,
            threads=2,
            hash_mb=256,
        )

        log.info("Starting Maia engine (lc0 + maia-1100)...")
        self.maia_engine = MaiaEngine()
        self.maia_engine.start()
        log.info("Maia engine ready")

    def _stop_engines(self):
        """Shut down all engines."""
        if self.sf_pool:
            self.sf_pool.shutdown()
        if self.maia_engine:
            self.maia_engine.stop()
        log.info("All engines stopped")

    def _event_loop_with_reconnect(self):
        """Event loop with automatic reconnection."""
        backoff = 1
        while True:
            try:
                self._event_loop()
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.warning(f"Event stream disconnected: {e}")
                log.info(f"Reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                log.info("Reconnecting to event stream...")

    def _event_loop(self):
        """Main event loop -- process challenges, game starts, etc."""
        log.info(f"Listening for incoming events (Stonefish {self.elo_target})...")
        for event in self.client.bots.stream_incoming_events():
            event_type = event.get("type", "")
            if event_type == "challenge":
                self._handle_challenge(event.get("challenge", {}))
            elif event_type == "challengeCanceled":
                log.info("Challenge canceled")
            elif event_type == "gameStart":
                self._handle_game_start(event.get("game", {}))
            elif event_type == "gameFinish":
                self._handle_game_finish(event.get("game", {}))

    def _handle_challenge(self, challenge):
        """Accept or decline an incoming challenge."""
        challenge_id = challenge.get("id", "")
        challenger = challenge.get("challenger", {}).get("id", "?")
        variant = challenge.get("variant", {}).get("key", "standard")
        rated = challenge.get("rated", False)
        speed = challenge.get("speed", "?")
        time_control = challenge.get("timeControl", {})
        tc_limit = time_control.get("limit", 0)

        log.info(f"Challenge from {challenger}: {variant} {speed} rated={rated} limit={tc_limit}s")

        # Check variant
        if variant not in self.SUPPORTED_VARIANTS:
            log.info(f"Declining: unsupported variant {variant}")
            self._decline(challenge_id, "variant")
            return

        # Check time control
        if tc_limit < self.MIN_TIME_CONTROL:
            log.info(f"Declining: time control too fast ({tc_limit}s < {self.MIN_TIME_CONTROL}s)")
            self._decline(challenge_id, "timeControl")
            return

        # Check rated/casual preference
        if self.rated_only and not rated:
            log.info("Declining: casual (rated-only mode)")
            self._decline(challenge_id, "casual")
            return

        if self.casual_only and rated:
            log.info("Declining: rated (casual-only mode)")
            self._decline(challenge_id, "rated")
            return

        # Check concurrent game limit
        active_count = sum(1 for t in self.active_games.values() if t.is_alive())
        if active_count >= self.max_games:
            log.info(f"Declining: already playing {active_count} game(s)")
            self._decline(challenge_id, "later")
            return

        log.info(f"Accepting challenge from {challenger}")
        try:
            self.client.bots.accept_challenge(challenge_id)
        except berserk.exceptions.ResponseError as e:
            log.error(f"Failed to accept challenge: {e}")

    def _decline(self, challenge_id: str, reason: str):
        """Decline a challenge with a reason."""
        try:
            self.client.bots.decline_challenge(challenge_id, reason=reason)
        except Exception as e:
            log.warning(f"Failed to decline challenge: {e}")

    def _handle_game_start(self, game_info):
        """Spawn a game handler thread for a new game."""
        game_id = game_info.get("gameId", game_info.get("id", ""))
        if not game_id:
            return

        log.info(f"Game started: {game_id}")
        handler = GameHandler(
            client=self.client,
            game_id=game_id,
            bot_id=self.bot_id,
            elo_target=self.elo_target,
            sf_pool=self.sf_pool,
            maia_engine=self.maia_engine,
        )
        thread = threading.Thread(
            target=handler.run, name=f"game-{game_id}", daemon=True,
        )
        self.active_games[game_id] = thread
        thread.start()

    def _handle_game_finish(self, game_info):
        """Clean up after a game ends."""
        game_id = game_info.get("gameId", game_info.get("id", ""))
        log.info(f"Game finished: {game_id}")
        self.active_games.pop(game_id, None)

    def challenge_player(
        self,
        username: str,
        rated: bool = False,
        clock_limit: int = 600,
        clock_increment: int = 5,
    ):
        """Send a challenge to a player."""
        log.info(f"Challenging {username} ({clock_limit}s+{clock_increment}s, rated={rated})")
        try:
            self.client.challenges.create(
                username,
                rated=rated,
                clock_limit=clock_limit,
                clock_increment=clock_increment,
            )
            log.info("Challenge sent!")
        except berserk.exceptions.ResponseError as e:
            log.error(f"Failed to challenge {username}: {e}")


# ---------------------------------------------------------------------------
# Token loading and CLI
# ---------------------------------------------------------------------------

def load_token() -> str:
    """Load Lichess API token from env or file."""
    token = os.environ.get("LICHESS_TOKEN", "").strip()
    if token:
        return token

    token_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lichess.token"
    )
    if os.path.exists(token_path):
        with open(token_path) as f:
            token = f.read().strip()
        if token:
            return token

    print("ERROR: No Lichess API token found!")
    print()
    print("Either:")
    print("  1. Set the LICHESS_TOKEN environment variable")
    print("  2. Save your token to lichess.token in this directory")
    print()
    print("Get a token at: https://lichess.org/account/oauth/token")
    print("Required scopes: bot:play, challenge:read, challenge:write")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Stonefish Lichess Bot -- plays like a real low-ELO player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python lichess_bot.py --elo 500                    # listen for challenges\n"
            "  python lichess_bot.py --elo 700 --challenge USER   # challenge a player\n"
            "  python lichess_bot.py --elo 900 --clock 15+10      # custom time control\n"
        ),
    )
    parser.add_argument(
        "--elo", type=int, default=500, choices=[500, 700, 900],
        help="Bot target ELO persona (default: 500)",
    )
    parser.add_argument(
        "--challenge", type=str, default=None,
        help="Challenge a specific Lichess player",
    )
    parser.add_argument(
        "--clock", type=str, default="10+5",
        help="Time control for outgoing challenges (e.g. '15+10', default: '10+5')",
    )
    parser.add_argument(
        "--rated", action="store_true",
        help="Make outgoing challenges rated",
    )
    parser.add_argument(
        "--rated-only", action="store_true",
        help="Only accept rated incoming challenges",
    )
    parser.add_argument(
        "--casual-only", action="store_true",
        help="Only accept casual incoming challenges",
    )
    parser.add_argument(
        "--max-games", type=int, default=3,
        help="Max concurrent games (default: 3)",
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print(f"   STONEFISH {args.elo} -- Lichess Bot")
    print("=" * 55)
    print()

    token = load_token()
    bot = StonefishLichessBot(
        token=token,
        elo_target=args.elo,
        max_games=args.max_games,
        rated_only=args.rated_only,
        casual_only=args.casual_only,
    )

    if args.challenge:
        # Challenge mode: start engines, challenge, listen
        bot.bot_id = bot.client.account.get().get("id", "")
        log.info(f"Logged in as: {bot.bot_id}")
        bot._start_engines()

        parts = args.clock.split("+")
        clock_limit = int(parts[0]) * 60 if not parts[0].isdigit() or int(parts[0]) < 60 else int(parts[0])
        # Handle both "10+5" (minutes) and "600+5" (seconds) formats
        if int(parts[0]) < 60:
            clock_limit = int(parts[0]) * 60
        else:
            clock_limit = int(parts[0])
        clock_increment = int(parts[1]) if len(parts) > 1 else 0

        bot.challenge_player(
            args.challenge,
            rated=args.rated,
            clock_limit=clock_limit,
            clock_increment=clock_increment,
        )
        try:
            bot._event_loop()
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            bot._stop_engines()
    else:
        # Listen mode
        bot.start()


if __name__ == "__main__":
    main()
