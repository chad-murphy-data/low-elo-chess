# Chess Blunder Hazard Model

Research project exploring blunder prediction in low-ELO chess (500–1500 Lichess).

Builds a dataset of per-move features from Lichess blitz games and fits logistic regression models to predict blunder probability as a function of player ELO, position complexity, clock time, and piece recency.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python run_pipeline.py
```

Or run individual steps:

```bash
# Step 1: Collect games from Lichess API (snowball sampling by ELO band)
python run_pipeline.py collect

# Step 2: Parse PGNs into per-move feature CSV
python run_pipeline.py features

# Step 3: Run logistic regression + hypothesis tests
python run_pipeline.py analyze

# Step 4: Generate plots
python run_pipeline.py plot
```

### Options

```
--data-dir DIR        Data directory (default: data/)
--results-dir DIR     Results directory (default: results/)
--target-users N      Users per ELO band (default: 200)
--target-games N      Games per user (default: 20)
--max-iterations N    Max snowball iterations (default: 50)
```

For a quick test run:

```bash
python run_pipeline.py collect --target-users 10 --target-games 5 --max-iterations 5
python run_pipeline.py features analyze plot
```

## Output

- `data/moves_features.csv` — Per-move dataset with all features
- `results/model_summary.txt` — Full regression output with coefficients and p-values
- `results/findings.md` — Human-readable summary of key findings
- `results/figures/` — All plots (PNG)

## Research Questions

1. What position and player characteristics predict blunder probability?
2. Does position complexity (num legal moves) moderate blunder rate beyond ELO?
3. Does piece recency predict hung pieces or missed captures?
4. Does move distance vary by ELO band?
5. At what rate do players punish hanging pieces by ELO band?

## Data Source

[Lichess API](https://lichess.org/api) — games include Stockfish evaluations (`[%eval]`) and clock times (`[%clk]`).
