"""
Play Against Stonefish — Browser UI
====================================
A browser-based interface to play chess against the Stonefish low-ELO bot.

Run:  python play_browser.py [--elo 500] [--color white] [--port 5002]
"""

import argparse
import json
import os
import sys
import threading
import webbrowser

import chess

from flask import Flask, Response, request, jsonify, render_template_string

from src.bot import StonefishBot
from src.game_state import GameState
from src.game_logger import GameLogger

# Force unbuffered stdout
if os.environ.get("PYTHONUNBUFFERED") != "1":
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

game_lock = threading.Lock()
game = None  # GameSession instance


class GameSession:
    """Holds one game session."""

    def __init__(self, elo, human_color_str):
        self.elo = elo
        self.human_is_white = human_color_str == "white"
        bot_color = chess.BLACK if self.human_is_white else chess.WHITE

        self.game_state = GameState(bot_color=bot_color)
        self.bot = StonefishBot(elo_target=elo)
        self.bot.start_engines()

        self.logger = GameLogger()
        self.logger.open()

        bot_color_str = "white" if bot_color == chess.WHITE else "black"
        self.game_id = self.logger.start_game(
            bot_elo=elo,
            bot_color=bot_color_str,
            opponent_type="human_browser",
        )

        self.game_over = False
        self.result_str = ""
        self.move_count = 0

        print(f"\nNew game: Human={human_color_str}, ELO={elo}")

    def is_human_turn(self):
        if self.game_over:
            return False
        return not self.game_state.is_bot_turn

    def make_human_move(self, uci_str):
        """Process a human move. Returns dict with game state update."""
        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            return {"error": "Invalid move format"}

        if move not in self.game_state.board.legal_moves:
            return {"error": "Illegal move"}

        san = self.game_state.board.san(move)
        eval_before = self.game_state.get_eval()

        # Push human move
        record = self.game_state.push_move(move, eval_before=eval_before)

        # Evaluate after human move
        eval_after = self.bot.evaluate_position(self.game_state)
        record.eval_after = eval_after

        # Compute cp_loss for human
        human_color = chess.WHITE if self.human_is_white else chess.BLACK
        cp_loss = None
        if eval_before is not None and eval_after is not None:
            if human_color == chess.WHITE:
                cp_loss = max(0, eval_before - eval_after)
            else:
                cp_loss = max(0, eval_after - eval_before)

        was_blunder = cp_loss is not None and cp_loss > 100

        self.logger.log_move(
            game_id=self.game_id,
            move_number=self.game_state.move_number,
            ply=self.game_state.ply,
            player="human",
            fen_before=self.game_state.board.fen(),
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

        self.move_count += 1
        print(f"  Human: {san}")

        result = {
            "fen": self.game_state.board.fen(),
            "san": san,
            "eval": eval_after,
        }

        # Check game over
        if self.game_state.is_game_over:
            result.update(self._handle_game_over())
        else:
            # Bot replies
            sf_data = self._make_bot_move()
            result["bot_move"] = sf_data

        return result

    def make_bot_first_move(self):
        """If bot is white, make its first move."""
        if self.human_is_white:
            return None
        # Initial eval
        self.bot.evaluate_position(self.game_state)
        return self._make_bot_move()

    def _make_bot_move(self):
        """Bot thinks and plays."""
        eval_before = self.game_state.get_eval()

        move, metadata = self.bot.select_move(self.game_state)
        san = self.game_state.board.san(move)

        # Push bot move
        record = self.game_state.push_move(move, eval_before=eval_before)

        # Evaluate after
        eval_after = self.bot.evaluate_position(self.game_state)
        record.eval_after = eval_after

        # Compute cp_loss for bot
        bot_color = chess.BLACK if self.human_is_white else chess.WHITE
        cp_loss = None
        if eval_before is not None and eval_after is not None:
            if bot_color == chess.WHITE:
                cp_loss = max(0, eval_before - eval_after)
            else:
                cp_loss = max(0, eval_after - eval_before)

        was_blunder = cp_loss is not None and cp_loss > 100

        self.logger.log_move(
            game_id=self.game_id,
            move_number=self.game_state.move_number,
            ply=self.game_state.ply,
            player="bot",
            fen_before=self.game_state.board.fen(),
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

        self.move_count += 1

        mechanism = metadata["mechanism"]
        maia_rank = metadata.get("maia_rank", "?")
        blunder_tag = ""
        if "blunder" in mechanism:
            mag = metadata.get("blunder_magnitude", "?")
            blunder_tag = f"  ** BLUNDER ({mag}) **"
        print(f"  Bot: {san} [rank {maia_rank}] {mechanism}{blunder_tag}")

        data = {
            "fen": self.game_state.board.fen(),
            "move": move.uci(),
            "san": san,
            "eval": eval_after,
            "mechanism": mechanism,
            "maia_rank": maia_rank,
            "was_blunder": "blunder" in mechanism,
        }

        if self.game_state.is_game_over:
            data["game_over"] = self._handle_game_over()

        return data

    def _handle_game_over(self):
        self.game_over = True
        outcome = self.game_state.board.outcome()
        self.result_str = outcome.result() if outcome else "*"

        who_won = "draw"
        if self.result_str == "1-0":
            who_won = "white"
        elif self.result_str == "0-1":
            who_won = "black"

        human_won = (who_won == "white" and self.human_is_white) or \
                    (who_won == "black" and not self.human_is_white)

        reason = ""
        if self.game_state.board.is_checkmate():
            reason = "checkmate"
        elif self.game_state.board.is_stalemate():
            reason = "stalemate"
        elif self.game_state.board.is_insufficient_material():
            reason = "insufficient material"
        elif self.game_state.board.is_fifty_moves():
            reason = "50-move rule"
        elif self.game_state.board.is_repetition():
            reason = "threefold repetition"

        self.logger.end_game(
            self.game_id, result=self.result_str,
            total_moves=self.game_state.move_number,
        )

        print(f"\n  Game over: {self.result_str} ({reason})")

        return {
            "game_over": True,
            "result": self.result_str,
            "human_won": human_won,
            "draw": who_won == "draw",
            "reason": reason,
        }

    def get_legal_moves(self):
        return [m.uci() for m in self.game_state.board.legal_moves]

    def shutdown(self):
        try:
            self.bot.stop_engines()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global game
    data = request.json or {}
    elo = data.get("elo", 500)
    color = data.get("color", "white")

    with game_lock:
        if game is not None:
            game.shutdown()
        game = GameSession(elo, color)

        # Initial eval
        eval_cp = game.bot.evaluate_position(game.game_state)

        result = {
            "fen": game.game_state.board.fen(),
            "human_is_white": game.human_is_white,
            "elo": elo,
            "eval": eval_cp,
        }

        # If bot goes first
        if not game.human_is_white:
            bot_data = game.make_bot_first_move()
            if bot_data:
                result["bot_move"] = bot_data
                result["fen"] = bot_data["fen"]
                result["eval"] = bot_data["eval"]

    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def make_move():
    global game
    data = request.json or {}
    uci = data.get("move", "")

    with game_lock:
        if game is None:
            return jsonify({"error": "No active game"}), 400
        if game.game_over:
            return jsonify({"error": "Game is over"}), 400
        if not game.is_human_turn():
            return jsonify({"error": "Not your turn"}), 400

        result = game.make_human_move(uci)

    return jsonify(result)


@app.route("/api/legal_moves")
def legal_moves():
    with game_lock:
        if game is None:
            return jsonify({"moves": []})
        return jsonify({"moves": game.get_legal_moves()})


@app.route("/api/state")
def get_state():
    with game_lock:
        if game is None:
            return jsonify({"active": False})
        return jsonify({
            "active": True,
            "fen": game.game_state.board.fen(),
            "human_is_white": game.human_is_white,
            "is_human_turn": game.is_human_turn(),
            "game_over": game.game_over,
        })


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Play Against Stonefish</title>
<link rel="stylesheet"
      href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    display: flex;
    justify-content: center;
    padding: 20px;
    min-height: 100vh;
  }
  .container {
    display: flex;
    gap: 24px;
    max-width: 1100px;
    width: 100%;
  }
  .board-col { flex: 0 0 480px; }
  .info-col { flex: 1; min-width: 300px; display: flex; flex-direction: column; gap: 12px; }
  #board { width: 480px; }

  h1 { font-size: 1.4rem; color: #58a6ff; font-weight: 700; margin-bottom: 8px; }

  .panel {
    background: #161b22;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #30363d;
  }
  .panel h2 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
    margin-bottom: 8px;
  }

  /* Setup */
  .setup-row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .setup-row label { font-size: 0.85rem; color: #8b949e; }
  .setup-row select, .setup-row button {
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #0d1117;
    color: #c9d1d9;
    font-size: 0.9rem;
    cursor: pointer;
  }
  .setup-row button {
    background: #238636;
    border-color: #238636;
    color: #fff;
    font-weight: 700;
  }
  .setup-row button:hover { background: #2ea043; }

  /* Eval bar */
  .eval-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0;
  }
  .eval-bar-bg {
    flex: 1;
    height: 20px;
    background: #333;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .eval-bar-fill {
    height: 100%;
    background: #f0f0f0;
    transition: width 0.4s ease;
    border-radius: 4px 0 0 4px;
  }
  .eval-text {
    font-size: 0.85rem;
    font-weight: 700;
    min-width: 60px;
    text-align: right;
  }

  /* Move log */
  .move-log {
    flex: 1;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    background: #0d1117;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
  }
  .move-entry {
    animation: fadeIn 0.3s;
  }
  .move-num { color: #8b949e; }
  .move-san { color: #c9d1d9; }
  .move-san.bot { color: #58a6ff; }
  .move-mechanism { color: #8b949e; font-size: 0.75rem; }
  .move-blunder { color: #f85149; font-weight: 700; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  /* Status */
  #status {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 4px;
    min-height: 1.2em;
  }
  #thinking {
    display: none;
    color: #f59e0b;
    font-size: 0.85rem;
    margin-top: 4px;
  }

  /* Highlight squares */
  .highlight-legal {
    background: radial-gradient(circle, rgba(88,166,255,0.35) 25%, transparent 25%);
  }
  .highlight-last-move {
    background-color: rgba(255, 255, 0, 0.15) !important;
  }
</style>
</head>
<body>
<div class="container">
  <div class="board-col">
    <h1>Stonefish</h1>
    <div id="board"></div>
    <div class="eval-container">
      <div class="eval-bar-bg"><div class="eval-bar-fill" id="eval-bar" style="width:50%"></div></div>
      <div class="eval-text" id="eval-text">0.0</div>
    </div>
    <div id="status">Set up a new game to start.</div>
    <div id="thinking">Stonefish is thinking...</div>
  </div>
  <div class="info-col">
    <!-- Setup -->
    <div class="panel">
      <h2>New Game</h2>
      <div class="setup-row">
        <label>ELO:</label>
        <select id="elo-select">
          <option value="500" selected>500</option>
          <option value="700">700</option>
          <option value="900">900</option>
        </select>
        <label>Play as:</label>
        <select id="color-select">
          <option value="white" selected>White</option>
          <option value="black">Black</option>
          <option value="random">Random</option>
        </select>
        <button onclick="startGame()">New Game</button>
      </div>
    </div>

    <!-- Move log -->
    <div class="panel" style="flex:1; display:flex; flex-direction:column;">
      <h2>Moves</h2>
      <div class="move-log" id="move-log"></div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
// State
var gameActive = false;
var humanIsWhite = true;
var isHumanTurn = false;
var legalMoves = [];
var moveNum = 1;
var lastMoveFrom = null;
var lastMoveTo = null;

// Board setup
var boardConfig = {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  draggable: true,
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
  onMouseoutSquare: onMouseoutSquare,
  onMouseoverSquare: onMouseoverSquare,
  appearSpeed: 200,
  moveSpeed: 200,
};
var board = Chessboard('board', boardConfig);

// ---------------------------------------------------------------------------
// Game management
// ---------------------------------------------------------------------------
function startGame() {
  var elo = parseInt(document.getElementById('elo-select').value);
  var color = document.getElementById('color-select').value;
  if (color === 'random') {
    color = Math.random() < 0.5 ? 'white' : 'black';
  }

  moveNum = 1;
  document.getElementById('move-log').innerHTML = '';
  document.getElementById('status').textContent = 'Starting engines...';
  document.getElementById('thinking').style.display = 'none';
  updateEvalBar(null);

  $.ajax({
    url: '/api/new_game',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({elo: elo, color: color}),
    success: function(data) {
      gameActive = true;
      humanIsWhite = data.human_is_white;
      board.orientation(humanIsWhite ? 'white' : 'black');
      board.position(data.fen, false);
      updateEvalBar(data.eval);

      // If bot moved first
      if (data.bot_move) {
        logBotMove(data.bot_move);
        updateEvalBar(data.bot_move.eval);
        highlightSquares(data.bot_move.move);
        if (data.bot_move.game_over) {
          handleGameOver(data.bot_move.game_over);
          return;
        }
      }

      isHumanTurn = true;
      fetchLegalMoves();
      document.getElementById('status').textContent =
        'Playing as ' + (humanIsWhite ? 'White' : 'Black') +
        ' vs Stonefish ' + elo + '. Your turn!';
    },
    error: function() {
      document.getElementById('status').textContent = 'Error starting game.';
    }
  });
}

function fetchLegalMoves() {
  $.get('/api/legal_moves', function(data) {
    legalMoves = data.moves || [];
  });
}

// ---------------------------------------------------------------------------
// Move handling
// ---------------------------------------------------------------------------
function onDragStart(source, piece) {
  if (!gameActive || !isHumanTurn) return false;
  if (humanIsWhite && piece.search(/^b/) !== -1) return false;
  if (!humanIsWhite && piece.search(/^w/) !== -1) return false;
  return true;
}

function onDrop(source, target, piece) {
  if (!gameActive || !isHumanTurn) return 'snapback';

  var uci = source + target;
  // Auto-promote to queen
  if (piece === 'wP' && target[1] === '8') uci += 'q';
  if (piece === 'bP' && target[1] === '1') uci += 'q';

  var isLegal = legalMoves.some(function(m) {
    return m === uci || m.substring(0, 4) === uci.substring(0, 4);
  });
  if (!isLegal) return 'snapback';

  // Use exact legal move UCI
  var exactMove = legalMoves.find(function(m) {
    return m.substring(0, 4) === uci.substring(0, 4);
  });
  if (exactMove && exactMove.length > 4 && uci.length <= 4) {
    uci = exactMove;
  }

  sendMove(uci, source, target);
  return undefined;
}

function onSnapEnd() {}

function onMouseoverSquare(square) {
  if (!gameActive || !isHumanTurn) return;
  var targets = legalMoves.filter(function(m) { return m.substring(0, 2) === square; });
  if (targets.length === 0) return;
  targets.forEach(function(m) {
    var dest = m.substring(2, 4);
    $('#board .square-' + dest).addClass('highlight-legal');
  });
}

function onMouseoutSquare() {
  $('#board .square-55d63').removeClass('highlight-legal');
}

function sendMove(uci, fromSq, toSq) {
  isHumanTurn = false;
  clearHighlights();
  document.getElementById('status').textContent = '';
  document.getElementById('thinking').style.display = 'block';

  $.ajax({
    url: '/api/move',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({move: uci}),
    success: function(data) {
      document.getElementById('thinking').style.display = 'none';

      if (data.error) {
        document.getElementById('status').textContent = 'Error: ' + data.error;
        isHumanTurn = true;
        return;
      }

      // Update after human move
      board.position(data.fen, true);
      updateEvalBar(data.eval);
      logHumanMove(data.san, fromSq, toSq);

      // Check game over after human move
      if (data.game_over) {
        handleGameOver(data);
        return;
      }

      // Process bot response
      if (data.bot_move) {
        board.position(data.bot_move.fen, true);
        updateEvalBar(data.bot_move.eval);
        logBotMove(data.bot_move);
        highlightSquares(data.bot_move.move);

        if (data.bot_move.game_over) {
          handleGameOver(data.bot_move.game_over);
          return;
        }

        isHumanTurn = true;
        fetchLegalMoves();
        document.getElementById('status').textContent = 'Your turn!';
      }
    },
    error: function() {
      document.getElementById('thinking').style.display = 'none';
      document.getElementById('status').textContent = 'Error sending move.';
      isHumanTurn = true;
    }
  });
}

// ---------------------------------------------------------------------------
// Eval bar
// ---------------------------------------------------------------------------
function updateEvalBar(evalCp) {
  var bar = document.getElementById('eval-bar');
  var text = document.getElementById('eval-text');

  if (evalCp === null || evalCp === undefined) {
    bar.style.width = '50%';
    text.textContent = '0.0';
    text.style.color = '#c9d1d9';
    return;
  }

  // Clamp to [-500, +500] for display
  var clamped = Math.max(-500, Math.min(500, evalCp));
  var pct = ((clamped + 500) / 1000) * 100;
  bar.style.width = pct + '%';

  var pawns = evalCp / 100.0;
  if (Math.abs(evalCp) >= 9000) {
    text.textContent = evalCp > 0 ? 'M+' : 'M-';
  } else {
    text.textContent = (pawns >= 0 ? '+' : '') + pawns.toFixed(1);
  }

  if (pawns > 0.5) text.style.color = '#f0f0f0';
  else if (pawns < -0.5) text.style.color = '#8b949e';
  else text.style.color = '#c9d1d9';
}

// ---------------------------------------------------------------------------
// Move log
// ---------------------------------------------------------------------------
function logHumanMove(san) {
  var log = document.getElementById('move-log');

  // If human is White, start a new move number line
  if (humanIsWhite) {
    var entry = document.createElement('div');
    entry.className = 'move-entry';
    entry.id = 'move-' + moveNum;
    entry.innerHTML =
      '<span class="move-num">' + moveNum + '.</span> ' +
      '<span class="move-san">' + san + '</span>';
    log.appendChild(entry);
  } else {
    // Human is Black — append to existing move line
    var entry = document.getElementById('move-' + moveNum);
    if (entry) {
      entry.innerHTML += '  <span class="move-san">' + san + '</span>';
      moveNum++;
    } else {
      // Fallback: create new line
      var newEntry = document.createElement('div');
      newEntry.className = 'move-entry';
      newEntry.id = 'move-' + moveNum;
      newEntry.innerHTML =
        '<span class="move-num">' + moveNum + '.</span> ... ' +
        '<span class="move-san">' + san + '</span>';
      log.appendChild(newEntry);
      moveNum++;
    }
  }
  log.scrollTop = log.scrollHeight;
}

function logBotMove(data) {
  var log = document.getElementById('move-log');
  var san = data.san;
  var mechanism = data.mechanism || '';
  var isBlunder = data.was_blunder;
  var rank = data.maia_rank;

  var mechText = '';
  if (mechanism) {
    var label = mechanism.replace(/_/g, ' ');
    if (rank) label = 'rank ' + rank + ', ' + label;
    mechText = ' <span class="move-mechanism">[' + label + ']</span>';
    if (isBlunder) {
      mechText = ' <span class="move-blunder">??</span>' + mechText;
    }
  }

  if (!humanIsWhite) {
    // Bot is White — start a new line
    var entry = document.createElement('div');
    entry.className = 'move-entry';
    entry.id = 'move-' + moveNum;
    entry.innerHTML =
      '<span class="move-num">' + moveNum + '.</span> ' +
      '<span class="move-san bot">' + san + '</span>' + mechText;
    log.appendChild(entry);
  } else {
    // Bot is Black — append to existing line
    var entry = document.getElementById('move-' + moveNum);
    if (entry) {
      entry.innerHTML +=
        '  <span class="move-san bot">' + san + '</span>' + mechText;
      moveNum++;
    } else {
      var newEntry = document.createElement('div');
      newEntry.className = 'move-entry';
      newEntry.id = 'move-' + moveNum;
      newEntry.innerHTML =
        '<span class="move-num">' + moveNum + '.</span> ... ' +
        '<span class="move-san bot">' + san + '</span>' + mechText;
      log.appendChild(newEntry);
      moveNum++;
    }
  }
  log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------------------
// Game over
// ---------------------------------------------------------------------------
function handleGameOver(data) {
  gameActive = false;
  isHumanTurn = false;
  document.getElementById('thinking').style.display = 'none';

  var msg = 'Game over: ' + data.result;
  if (data.human_won) msg = 'You won! (' + data.reason + ')';
  else if (data.draw) msg = 'Draw (' + data.reason + ')';
  else msg = 'You lost (' + data.reason + ')';

  document.getElementById('status').textContent = msg;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function highlightSquares(uci) {
  clearHighlights();
  if (uci && uci.length >= 4) {
    var from = uci.substring(0, 2);
    var to = uci.substring(2, 4);
    $('#board .square-' + from).addClass('highlight-last-move');
    $('#board .square-' + to).addClass('highlight-last-move');
  }
}

function clearHighlights() {
  $('#board .square-55d63').removeClass('highlight-legal highlight-last-move');
}

// Start a game automatically on load
$(function() {
  startGame();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play Against Stonefish (Browser)")
    parser.add_argument("--elo", type=int, default=500, choices=[500, 700, 900],
                        help="Bot ELO (default: 500)")
    parser.add_argument("--color", choices=["white", "black", "random"],
                        default="white", help="Your color (default: white)")
    parser.add_argument("--port", type=int, default=5002,
                        help="Web server port (default: 5002)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Stonefish — Play in Browser")
    print("=" * 50)
    print(f"  Default ELO: {args.elo}")
    print(f"  Default color: {args.color}")
    print(f"  URL: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    threading.Timer(1.5, lambda: webbrowser.open(
        f"http://localhost:{args.port}")).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
