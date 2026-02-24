"""
Watch Stonefish vs Stonefish — Browser UI
==========================================
Watch two Stonefish bots play each other with a 2-second delay between moves.

Run:  python watch_browser.py [--white-elo 500] [--black-elo 500] [--delay 2] [--port 5003]
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser

import chess

from flask import Flask, request, jsonify, render_template_string

from src.bot import StonefishBot
from src.game_state import GameState
from src.game_logger import GameLogger

if os.environ.get("PYTHONUNBUFFERED") != "1":
    sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)

watch_lock = threading.Lock()
watch = None  # WatchSession instance


class WatchSession:
    """Two bots playing each other, one move at a time."""

    def __init__(self, white_elo, black_elo):
        self.white_elo = white_elo
        self.black_elo = black_elo

        # White bot's game state (it thinks it's white)
        self.white_state = GameState(bot_color=chess.WHITE)
        self.white_bot = StonefishBot(elo_target=white_elo)

        # Black bot's game state (it thinks it's black)
        self.black_state = GameState(bot_color=chess.BLACK)
        self.black_bot = StonefishBot(elo_target=black_elo)

        self.white_bot.start_engines()
        self.black_bot.start_engines()

        self.logger = GameLogger()
        self.logger.open()
        self.game_id = self.logger.start_game(
            bot_elo=white_elo,
            bot_color="both",
            opponent_type=f"bot_{black_elo}",
        )

        self.move_history = []  # list of dicts with move info
        self.game_over = False
        self.result_str = ""
        self.result_reason = ""

        # Initial eval
        self.last_eval = self.white_bot.evaluate_position(self.white_state)

        print(f"\nWatch: {white_elo} (White) vs {black_elo} (Black)")

    @property
    def board(self):
        """Both states share the same position; use white's as canonical."""
        return self.white_state.board

    def make_next_move(self):
        """Have the current side make a move. Returns move data dict."""
        if self.game_over:
            return {"game_over": True, "result": self.result_str, "reason": self.result_reason}

        is_white_turn = self.board.turn == chess.WHITE
        bot = self.white_bot if is_white_turn else self.black_bot
        state = self.white_state if is_white_turn else self.black_state
        other_state = self.black_state if is_white_turn else self.white_state
        elo = self.white_elo if is_white_turn else self.black_elo
        side = "White" if is_white_turn else "Black"

        eval_before = state.get_eval()

        move, metadata = bot.select_move(state)
        san = state.board.san(move)
        is_capture = state.board.is_capture(move)

        # Push to active bot's state
        record = state.push_move(move, eval_before=eval_before)

        # Push to opponent's state too (so it tracks the move)
        other_state.push_move(move, eval_before=other_state.get_eval())

        # Evaluate after (using the bot that just moved)
        eval_after = bot.evaluate_position(state)
        if eval_after is not None:
            other_state.set_eval(eval_after)
        self.last_eval = eval_after

        mechanism = metadata["mechanism"]
        maia_rank = metadata.get("maia_rank", "?")
        is_blunder = "blunder" in mechanism

        blunder_tag = ""
        if is_blunder:
            mag = metadata.get("blunder_magnitude", "?")
            blunder_tag = f"  ** BLUNDER ({mag}) **"

        print(f"  {side} ({elo}): {san} [rank {maia_rank}] {mechanism}{blunder_tag}")

        # Compute cp_loss
        bot_color = chess.WHITE if is_white_turn else chess.BLACK
        cp_loss = None
        if eval_before is not None and eval_after is not None:
            if bot_color == chess.WHITE:
                cp_loss = max(0, eval_before - eval_after)
            else:
                cp_loss = max(0, eval_after - eval_before)

        self.logger.log_move(
            game_id=self.game_id,
            move_number=state.move_number,
            ply=len(self.move_history) + 1,
            player=f"bot_{elo}",
            fen_before=state.board.fen(),
            move_uci=move.uci(),
            move_san=san,
            piece_moved=record.piece_name,
            is_capture=is_capture,
            is_check=record.is_check,
            eval_before=eval_before,
            eval_after=eval_after,
            cp_loss=cp_loss,
            was_blunder=is_blunder,
            mechanism=mechanism,
            maia_rank=maia_rank,
            maia_top5=json.dumps(metadata.get("maia_top5", [])),
            notes=metadata.get("notes"),
        )

        move_data = {
            "fen": state.board.fen(),
            "move": move.uci(),
            "san": san,
            "side": side,
            "elo": elo,
            "eval": eval_after,
            "mechanism": mechanism,
            "maia_rank": maia_rank,
            "is_blunder": is_blunder,
            "blunder_magnitude": metadata.get("blunder_magnitude"),
            "move_number": state.move_number,
        }

        self.move_history.append(move_data)

        # Check game over
        if state.is_game_over:
            self.game_over = True
            outcome = state.board.outcome()
            self.result_str = outcome.result() if outcome else "*"

            if state.board.is_checkmate():
                self.result_reason = "checkmate"
            elif state.board.is_stalemate():
                self.result_reason = "stalemate"
            elif state.board.is_insufficient_material():
                self.result_reason = "insufficient material"
            elif state.board.is_fifty_moves():
                self.result_reason = "50-move rule"
            elif state.board.is_repetition():
                self.result_reason = "threefold repetition"

            self.logger.end_game(
                self.game_id, result=self.result_str,
                total_moves=state.move_number,
            )
            print(f"\n  Game over: {self.result_str} ({self.result_reason})")

            move_data["game_over"] = True
            move_data["result"] = self.result_str
            move_data["reason"] = self.result_reason

        return move_data

    def shutdown(self):
        try:
            self.white_bot.stop_engines()
        except Exception:
            pass
        try:
            self.black_bot.stop_engines()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global watch
    data = request.json or {}
    white_elo = data.get("white_elo", 500)
    black_elo = data.get("black_elo", 500)

    with watch_lock:
        if watch is not None:
            watch.shutdown()
        watch = WatchSession(white_elo, black_elo)

    return jsonify({
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "white_elo": white_elo,
        "black_elo": black_elo,
        "eval": watch.last_eval,
    })


@app.route("/api/next_move", methods=["POST"])
def next_move():
    with watch_lock:
        if watch is None:
            return jsonify({"error": "No active game"}), 400
        if watch.game_over:
            return jsonify({
                "game_over": True,
                "result": watch.result_str,
                "reason": watch.result_reason,
            })
        data = watch.make_next_move()
    return jsonify(data)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Watch Stonefish vs Stonefish</title>
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
  .setup-row button.stop { background: #da3633; border-color: #da3633; }
  .setup-row button.stop:hover { background: #f85149; }

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
    max-height: 450px;
    overflow-y: auto;
    padding: 10px;
    background: #0d1117;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
  }
  .move-entry { animation: fadeIn 0.3s; }
  .move-num { color: #8b949e; }
  .move-san { color: #c9d1d9; }
  .move-san.white { color: #f0f0f0; }
  .move-san.black { color: #58a6ff; }
  .move-mechanism { color: #8b949e; font-size: 0.75rem; }
  .move-blunder { color: #f85149; font-weight: 700; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  #status {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 4px;
    min-height: 1.2em;
  }

  .highlight-last-move {
    background-color: rgba(255, 255, 0, 0.15) !important;
  }

  /* Player labels */
  .player-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 0.9rem;
  }
  .player-label .name { font-weight: 700; }
  .player-label .elo-badge {
    background: #30363d;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
  }
</style>
</head>
<body>
<div class="container">
  <div class="board-col">
    <h1>Stonefish vs Stonefish</h1>
    <div class="player-label">
      <span class="name" style="color:#58a6ff">Black</span>
      <span class="elo-badge" id="black-elo">500</span>
    </div>
    <div id="board"></div>
    <div class="player-label">
      <span class="name" style="color:#f0f0f0">White</span>
      <span class="elo-badge" id="white-elo">500</span>
    </div>
    <div class="eval-container">
      <div class="eval-bar-bg"><div class="eval-bar-fill" id="eval-bar" style="width:50%"></div></div>
      <div class="eval-text" id="eval-text">0.0</div>
    </div>
    <div id="status">Configure and start a game.</div>
  </div>
  <div class="info-col">
    <div class="panel">
      <h2>Setup</h2>
      <div class="setup-row">
        <label>White:</label>
        <select id="white-elo-select">
          <option value="500" selected>500</option>
          <option value="700">700</option>
          <option value="900">900</option>
        </select>
        <label>Black:</label>
        <select id="black-elo-select">
          <option value="500" selected>500</option>
          <option value="700">700</option>
          <option value="900">900</option>
        </select>
      </div>
      <div class="setup-row" style="margin-top:8px">
        <label>Delay:</label>
        <select id="delay-select">
          <option value="1">1s</option>
          <option value="2" selected>2s</option>
          <option value="3">3s</option>
          <option value="5">5s</option>
        </select>
        <button id="start-btn" onclick="startWatch()">Start Game</button>
        <button id="stop-btn" class="stop" onclick="stopWatch()" style="display:none">Stop</button>
      </div>
    </div>

    <div class="panel" style="flex:1; display:flex; flex-direction:column;">
      <h2>Moves</h2>
      <div class="move-log" id="move-log"></div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
var running = false;
var timer = null;
var moveNum = 1;
var waitingForWhite = true; // track whose move log line we're building

var board = Chessboard('board', {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  draggable: false,
  appearSpeed: 200,
  moveSpeed: 200,
});

function startWatch() {
  var whiteElo = parseInt(document.getElementById('white-elo-select').value);
  var blackElo = parseInt(document.getElementById('black-elo-select').value);

  document.getElementById('white-elo').textContent = whiteElo;
  document.getElementById('black-elo').textContent = blackElo;
  document.getElementById('move-log').innerHTML = '';
  document.getElementById('status').textContent = 'Starting engines...';
  document.getElementById('start-btn').style.display = 'none';
  document.getElementById('stop-btn').style.display = 'inline-block';
  moveNum = 1;
  waitingForWhite = true;
  updateEvalBar(null);

  $.ajax({
    url: '/api/new_game',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({white_elo: whiteElo, black_elo: blackElo}),
    success: function(data) {
      board.position(data.fen, false);
      updateEvalBar(data.eval);
      running = true;
      document.getElementById('status').textContent = 'Game in progress...';
      scheduleNextMove();
    },
    error: function() {
      document.getElementById('status').textContent = 'Error starting game.';
      resetButtons();
    }
  });
}

function stopWatch() {
  running = false;
  if (timer) { clearTimeout(timer); timer = null; }
  document.getElementById('status').textContent = 'Stopped.';
  resetButtons();
}

function resetButtons() {
  document.getElementById('start-btn').style.display = 'inline-block';
  document.getElementById('stop-btn').style.display = 'none';
}

function scheduleNextMove() {
  if (!running) return;
  var delay = parseInt(document.getElementById('delay-select').value) * 1000;
  timer = setTimeout(requestNextMove, delay);
}

function requestNextMove() {
  if (!running) return;

  $.ajax({
    url: '/api/next_move',
    method: 'POST',
    contentType: 'application/json',
    data: '{}',
    success: function(data) {
      if (data.error) {
        document.getElementById('status').textContent = 'Error: ' + data.error;
        return;
      }

      if (data.fen) {
        board.position(data.fen, true);
      }
      if (data.move) {
        highlightSquares(data.move);
      }
      updateEvalBar(data.eval);
      logMove(data);

      if (data.game_over) {
        running = false;
        var msg = data.result + ' (' + data.reason + ')';
        document.getElementById('status').textContent = 'Game over: ' + msg;
        resetButtons();
        return;
      }

      scheduleNextMove();
    },
    error: function() {
      document.getElementById('status').textContent = 'Error fetching move.';
      running = false;
      resetButtons();
    }
  });
}

function logMove(data) {
  var log = document.getElementById('move-log');
  var san = data.san;
  var mechanism = data.mechanism || '';
  var isBlunder = data.is_blunder;
  var rank = data.maia_rank;
  var side = data.side;
  var elo = data.elo;

  var mechText = '';
  if (mechanism) {
    var label = mechanism.replace(/_/g, ' ');
    if (rank) label = 'r' + rank + ' ' + label;
    mechText = ' <span class="move-mechanism">[' + label + ']</span>';
    if (isBlunder) {
      mechText = ' <span class="move-blunder">??</span>' + mechText;
    }
  }

  var sideClass = side === 'White' ? 'white' : 'black';

  if (side === 'White') {
    // Start new line
    var entry = document.createElement('div');
    entry.className = 'move-entry';
    entry.id = 'move-' + moveNum;
    entry.innerHTML =
      '<span class="move-num">' + moveNum + '.</span> ' +
      '<span class="move-san ' + sideClass + '">' + san + '</span>' + mechText;
    log.appendChild(entry);
  } else {
    // Append to existing line
    var entry = document.getElementById('move-' + moveNum);
    if (entry) {
      entry.innerHTML +=
        '  <span class="move-san ' + sideClass + '">' + san + '</span>' + mechText;
    }
    moveNum++;
  }
  log.scrollTop = log.scrollHeight;
}

function updateEvalBar(evalCp) {
  var bar = document.getElementById('eval-bar');
  var text = document.getElementById('eval-text');
  if (evalCp === null || evalCp === undefined) {
    bar.style.width = '50%';
    text.textContent = '0.0';
    text.style.color = '#c9d1d9';
    return;
  }
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

function highlightSquares(uci) {
  $('#board .square-55d63').removeClass('highlight-last-move');
  if (uci && uci.length >= 4) {
    var from = uci.substring(0, 2);
    var to = uci.substring(2, 4);
    $('#board .square-' + from).addClass('highlight-last-move');
    $('#board .square-' + to).addClass('highlight-last-move');
  }
}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Watch Stonefish vs Stonefish (Browser)")
    parser.add_argument("--white-elo", type=int, default=500, choices=[500, 700, 900])
    parser.add_argument("--black-elo", type=int, default=500, choices=[500, 700, 900])
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()

    print("=" * 50)
    print("  Stonefish vs Stonefish — Watch in Browser")
    print("=" * 50)
    print(f"  Default: {args.white_elo} (White) vs {args.black_elo} (Black)")
    print(f"  URL: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    threading.Timer(1.5, lambda: webbrowser.open(
        f"http://localhost:{args.port}")).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
