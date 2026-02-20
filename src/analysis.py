"""
Statistical analysis: logistic regression models and hypothesis tests.

Produces:
- Model comparisons (ELO-only baseline → full model)
- Coefficient tables with p-values
- AUC scores
- Specific hypothesis tests (complexity, recency, move distance)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_analysis_data(csv_path="data/moves_features.csv"):
    """Load and clean the dataset for analysis.

    Drops rows with missing evals, excludes near-mate positions,
    and adds derived columns.
    """
    df = pd.read_csv(csv_path)

    # Drop rows without eval data
    df = df.dropna(subset=["eval_before", "eval_after", "centipawn_loss"])

    # Exclude near-mate positions
    df = df[df["is_near_mate"] != True].copy()

    # Exclude rows where blunder label is missing
    df = df.dropna(subset=["is_blunder"])

    # Convert booleans
    bool_cols = [
        "is_blunder", "is_mistake", "is_inaccuracy",
        "in_check", "has_hanging_piece_before", "created_hanging_piece",
        "opponent_had_hanging_piece", "is_capture", "is_check_given",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool).astype(int)

    # Add ELO band column
    def elo_band(elo):
        if pd.isna(elo):
            return None
        elo = int(elo)
        for lo, hi in [(500, 700), (700, 900), (900, 1100),
                       (1100, 1300), (1300, 1500)]:
            if lo <= elo < hi:
                return f"{lo}-{hi}"
        if elo >= 1500:
            return "1500+"
        return "<500"

    df["elo_band"] = df["player_elo"].apply(elo_band)

    # Add interaction term
    df["elo_x_legal_moves"] = df["player_elo"] * df["num_legal_moves"]

    # Add clock bins
    def clock_bin(secs):
        if pd.isna(secs):
            return None
        if secs < 30:
            return "<30s"
        elif secs < 60:
            return "30-60s"
        elif secs < 120:
            return "1-2min"
        elif secs < 300:
            return "2-5min"
        else:
            return "5min+"

    df["clock_bin"] = df["clock_remaining"].apply(clock_bin)

    print(f"Analysis dataset: {len(df)} observations")
    print(f"  Blunder rate: {df['is_blunder'].mean():.3f}")
    print(f"  ELO bands: {df['elo_band'].value_counts().to_dict()}")

    return df


def run_logistic_models(df, results_dir="results"):
    """Run the three-model comparison: baseline → context → full.

    Returns a dict of model results and writes summary to file.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    target = "is_blunder"

    # Define feature sets
    baseline_features = ["player_elo"]
    context_features = baseline_features + ["clock_remaining", "move_number"]
    position_features = context_features + [
        "num_legal_moves", "piece_count", "material_balance",
        "move_distance", "in_check", "is_capture",
    ]
    full_features = position_features + ["moves_since_piece_last_moved"]
    interaction_features = full_features + ["elo_x_legal_moves"]

    models = {}
    output_lines = []

    output_lines.append("=" * 70)
    output_lines.append("BLUNDER PREDICTION: LOGISTIC REGRESSION RESULTS")
    output_lines.append("=" * 70)
    output_lines.append("")

    for name, features in [
        ("Model 1: ELO only", baseline_features),
        ("Model 2: ELO + context", context_features),
        ("Model 3: ELO + context + position", position_features),
        ("Model 4: Full model", full_features),
        ("Model 5: Full + ELO×complexity interaction", interaction_features),
    ]:
        output_lines.append(f"\n{'─' * 60}")
        output_lines.append(f"  {name}")
        output_lines.append(f"  Features: {features}")
        output_lines.append(f"{'─' * 60}")

        # Drop rows with missing values in the features we need
        subset = df.dropna(subset=features + [target])
        if len(subset) < 100:
            output_lines.append(f"  SKIPPED: only {len(subset)} valid observations")
            continue

        X = subset[features].astype(float)
        y = subset[target].astype(int)

        # Standardize for sklearn (not for statsmodels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- statsmodels for interpretability ---
        X_sm = sm.add_constant(X)
        try:
            logit_model = sm.Logit(y, X_sm)
            result = logit_model.fit(disp=0, maxiter=100)

            output_lines.append(f"\n  Observations: {len(y)}")
            output_lines.append(f"  Pseudo R²: {result.prsquared:.4f}")
            output_lines.append(f"  Log-Likelihood: {result.llf:.1f}")
            output_lines.append(f"  AIC: {result.aic:.1f}")
            output_lines.append(f"  BIC: {result.bic:.1f}")
            output_lines.append("")

            # Coefficient table
            coef_df = pd.DataFrame({
                "coef": result.params,
                "std_err": result.bse,
                "z": result.tvalues,
                "p_value": result.pvalues,
                "ci_low": result.conf_int()[0],
                "ci_high": result.conf_int()[1],
            })
            output_lines.append("  Coefficients:")
            output_lines.append(
                f"  {'Feature':<35} {'Coef':>10} {'Std Err':>10} "
                f"{'z':>8} {'P>|z|':>10} {'Sig':>5}"
            )
            output_lines.append("  " + "-" * 80)
            for feat, row in coef_df.iterrows():
                sig = ""
                if row["p_value"] < 0.001:
                    sig = "***"
                elif row["p_value"] < 0.01:
                    sig = "**"
                elif row["p_value"] < 0.05:
                    sig = "*"
                output_lines.append(
                    f"  {str(feat):<35} {row['coef']:>10.5f} "
                    f"{row['std_err']:>10.5f} {row['z']:>8.3f} "
                    f"{row['p_value']:>10.4f} {sig:>5}"
                )

        except Exception as e:
            output_lines.append(f"  statsmodels error: {e}")
            result = None

        # --- sklearn for AUC ---
        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X_scaled, y)
            y_prob = clf.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_prob)
            output_lines.append(f"\n  AUC (in-sample): {auc:.4f}")
        except Exception as e:
            output_lines.append(f"  sklearn AUC error: {e}")
            auc = None

        models[name] = {
            "result": result,
            "auc": auc,
            "n_obs": len(y),
            "features": features,
        }

    # Write summary
    summary_text = "\n".join(output_lines)
    summary_path = results_path / "model_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Model summary saved to {summary_path}")
    print(summary_text)

    return models


def run_hypothesis_tests(df, results_dir="results"):
    """Run the specific hypothesis tests from the analysis plan.

    Tests:
    1. Does complexity moderate blunder rate beyond ELO?
    2. Does piece recency predict hung pieces?
    3. Does recency predict missed captures?
    4. Does move distance vary by ELO band?
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("HYPOTHESIS TESTS")
    lines.append("=" * 70)

    # --- Test 1: Complexity × ELO interaction ---
    lines.append("\n\n--- Test 1: Does position complexity moderate blunder rate "
                 "beyond ELO? ---")
    subset = df.dropna(subset=["player_elo", "num_legal_moves", "is_blunder"])
    if len(subset) > 100:
        X = subset[["player_elo", "num_legal_moves", "elo_x_legal_moves"]].astype(float)
        X = sm.add_constant(X)
        y = subset["is_blunder"].astype(int)
        try:
            result = sm.Logit(y, X).fit(disp=0)
            lines.append(f"  N = {len(y)}")
            lines.append(f"  Pseudo R² = {result.prsquared:.4f}")
            for feat in ["player_elo", "num_legal_moves", "elo_x_legal_moves"]:
                coef = result.params[feat]
                pval = result.pvalues[feat]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                lines.append(f"  {feat}: coef={coef:.6f}, p={pval:.4f} ({sig})")
            lines.append("")
            if result.pvalues["elo_x_legal_moves"] < 0.05:
                direction = "negative" if result.params["elo_x_legal_moves"] < 0 else "positive"
                lines.append(f"  FINDING: Significant {direction} ELO×complexity interaction.")
                if direction == "negative":
                    lines.append("  → Low-ELO players are disproportionately hurt "
                                 "by complex positions.")
                else:
                    lines.append("  → Higher-ELO players are more affected by complexity "
                                 "(unexpected).")
            else:
                lines.append("  FINDING: No significant ELO×complexity interaction.")
        except Exception as e:
            lines.append(f"  Error: {e}")

    # --- Test 2: Piece recency predicts hung pieces ---
    lines.append("\n\n--- Test 2: Does piece recency predict hung piece creation? ---")
    hung_subset = df[df["created_hanging_piece"] == 1].dropna(
        subset=["hung_piece_recency", "centipawn_loss"]
    )
    if len(hung_subset) > 30:
        lines.append(f"  N (moves that created hanging pieces) = {len(hung_subset)}")
        corr = hung_subset["hung_piece_recency"].corr(hung_subset["centipawn_loss"])
        lines.append(f"  Correlation(hung_piece_recency, centipawn_loss) = {corr:.4f}")

        # Simple regression
        X = sm.add_constant(hung_subset[["hung_piece_recency"]].astype(float))
        y = hung_subset["centipawn_loss"].astype(float)
        try:
            ols = sm.OLS(y, X).fit()
            coef = ols.params["hung_piece_recency"]
            pval = ols.pvalues["hung_piece_recency"]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            lines.append(f"  OLS: coef={coef:.4f}, p={pval:.4f} ({sig})")
            if pval < 0.05 and coef > 0:
                lines.append("  FINDING: Higher recency → larger blunders when hanging pieces.")
                lines.append("  → Pieces out of the player's 'attentional model' get hung worse.")
            else:
                lines.append("  FINDING: Piece recency does not significantly predict "
                             "hung piece severity.")
        except Exception as e:
            lines.append(f"  Error: {e}")
    else:
        lines.append(f"  Insufficient data ({len(hung_subset)} obs)")

    # --- Test 3: Recency predicts missed captures ---
    lines.append("\n\n--- Test 3: Does recency predict missed captures? ---")
    opp_hanging = df[df["opponent_had_hanging_piece"] == 1].copy()
    if len(opp_hanging) > 30:
        opp_hanging["captured_it"] = opp_hanging["is_capture"].astype(int)
        opp_hanging_with_recency = opp_hanging.dropna(subset=["missed_capture_recency"])

        lines.append(f"  N (positions with opponent hanging piece) = {len(opp_hanging)}")
        lines.append(f"  Capture rate: {opp_hanging['captured_it'].mean():.3f}")

        if len(opp_hanging_with_recency) > 30:
            # Among those who didn't capture, what's the recency of missed pieces?
            missed = opp_hanging_with_recency[opp_hanging_with_recency["captured_it"] == 0]
            captured = opp_hanging_with_recency[opp_hanging_with_recency["captured_it"] == 1]

            if len(missed) > 5 and len(captured) > 5:
                lines.append(f"  Mean recency when captured: "
                             f"{captured['missed_capture_recency'].mean():.1f} plies (N/A — "
                             f"using opponent piece recency)")
                lines.append(f"  Mean recency when missed: "
                             f"{missed['missed_capture_recency'].mean():.1f} plies")

            # Logistic: does recency predict whether player captures?
            X = sm.add_constant(
                opp_hanging.dropna(subset=["missed_capture_recency"])[
                    ["missed_capture_recency", "player_elo"]
                ].astype(float)
            )
            y = opp_hanging.dropna(subset=["missed_capture_recency"])[
                "captured_it"
            ].astype(int)
            if len(y) > 30:
                try:
                    result = sm.Logit(y, X).fit(disp=0)
                    for feat in ["missed_capture_recency", "player_elo"]:
                        coef = result.params[feat]
                        pval = result.pvalues[feat]
                        sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                               else "*" if pval < 0.05 else "ns")
                        lines.append(f"  Logit({feat}→captured): coef={coef:.5f}, "
                                     f"p={pval:.4f} ({sig})")
                except Exception as e:
                    lines.append(f"  Error: {e}")
    else:
        lines.append(f"  Insufficient data ({len(opp_hanging)} obs)")

    # --- Test 4: Move distance by ELO band ---
    lines.append("\n\n--- Test 4: Does move distance vary by ELO band? ---")
    dist_data = df.dropna(subset=["move_distance", "elo_band"])
    if len(dist_data) > 100:
        dist_by_band = dist_data.groupby("elo_band")["move_distance"].agg(
            ["mean", "std", "count"]
        )
        lines.append("  Mean move distance by ELO band:")
        for band, row in dist_by_band.iterrows():
            lines.append(f"    {band}: mean={row['mean']:.2f}, "
                         f"std={row['std']:.2f}, n={int(row['count'])}")

        # Also: does move distance predict blunders controlling for ELO?
        subset = df.dropna(subset=["move_distance", "player_elo", "is_blunder"])
        if len(subset) > 100:
            X = sm.add_constant(subset[["move_distance", "player_elo"]].astype(float))
            y = subset["is_blunder"].astype(int)
            try:
                result = sm.Logit(y, X).fit(disp=0)
                coef = result.params["move_distance"]
                pval = result.pvalues["move_distance"]
                sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                       else "*" if pval < 0.05 else "ns")
                lines.append(f"\n  Logit(move_distance→blunder | ELO): "
                             f"coef={coef:.5f}, p={pval:.4f} ({sig})")
                if pval < 0.05 and coef > 0:
                    lines.append("  FINDING: Longer-distance moves are riskier "
                                 "after controlling for ELO.")
            except Exception as e:
                lines.append(f"  Error: {e}")

    # --- Descriptive: Blunder rate by ELO band ---
    lines.append("\n\n--- Descriptive: Blunder rate by ELO band ---")
    blunder_by_band = df.groupby("elo_band")["is_blunder"].agg(["mean", "count"])
    for band, row in blunder_by_band.iterrows():
        lines.append(f"  {band}: blunder_rate={row['mean']:.3f}, n={int(row['count'])}")

    # --- Descriptive: Blunder rate by move number ---
    lines.append("\n\n--- Descriptive: Blunder rate by game phase ---")
    df_with_mn = df.dropna(subset=["move_number"])
    if len(df_with_mn) > 0:
        df_with_mn = df_with_mn.copy()
        df_with_mn["phase"] = pd.cut(
            df_with_mn["move_number"],
            bins=[0, 20, 40, 60, 200],
            labels=["opening (1-10)", "early-mid (11-20)",
                    "mid-late (21-30)", "endgame (31+)"],
        )
        phase_blunder = df_with_mn.groupby("phase", observed=True)["is_blunder"].agg(
            ["mean", "count"]
        )
        for phase, row in phase_blunder.iterrows():
            lines.append(f"  {phase}: blunder_rate={row['mean']:.3f}, "
                         f"n={int(row['count'])}")

    # --- Descriptive: Hanging piece punishment rate by ELO ---
    lines.append("\n\n--- Descriptive: Hanging piece punishment rate by ELO band ---")
    opp_hanging = df[df["opponent_had_hanging_piece"] == 1].copy()
    if len(opp_hanging) > 0:
        opp_hanging["punished"] = opp_hanging["is_capture"].astype(int)
        punishment = opp_hanging.groupby("elo_band")["punished"].agg(["mean", "count"])
        for band, row in punishment.iterrows():
            lines.append(f"  {band}: punishment_rate={row['mean']:.3f}, "
                         f"n={int(row['count'])}")

    hypothesis_text = "\n".join(lines)

    # Append to model summary
    summary_path = results_path / "model_summary.txt"
    with open(summary_path, "a") as f:
        f.write("\n\n")
        f.write(hypothesis_text)
    print(hypothesis_text)

    return lines


def generate_findings(df, models, results_dir="results"):
    """Generate a human-readable summary of key findings."""
    results_path = Path(results_dir)
    lines = []

    lines.append("# Chess Blunder Hazard Model — Key Findings")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- **Total move observations:** {len(df):,}")

    n_games = df["game_id"].nunique()
    lines.append(f"- **Total games:** {n_games:,}")

    blunder_rate = df[df["is_blunder"].notna()]["is_blunder"].mean()
    lines.append(f"- **Overall blunder rate (CPL > 100):** {blunder_rate:.1%}")

    lines.append("")
    lines.append("### Blunder rate by ELO band")
    lines.append("")
    lines.append("| ELO Band | Blunder Rate | N |")
    lines.append("|----------|-------------|---|")
    for band in ["500-700", "700-900", "900-1100", "1100-1300", "1300-1500"]:
        band_df = df[df["elo_band"] == band]
        if len(band_df) > 0:
            rate = band_df["is_blunder"].mean()
            lines.append(f"| {band} | {rate:.1%} | {len(band_df):,} |")

    lines.append("")
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Features | Pseudo R² | AUC |")
    lines.append("|-------|----------|-----------|-----|")
    for name, info in models.items():
        if info.get("result") is not None:
            r2 = info["result"].prsquared
            auc = info.get("auc", "—")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
            lines.append(f"| {name} | {len(info['features'])} | "
                         f"{r2:.4f} | {auc_str} |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("*(See model_summary.txt for full statistical details)*")
    lines.append("")

    # Check if we have significant findings
    full_model = models.get("Model 4: Full model", {})
    if full_model.get("result") is not None:
        result = full_model["result"]
        significant_features = []
        for feat in result.params.index:
            if feat == "const":
                continue
            if result.pvalues[feat] < 0.05:
                direction = "+" if result.params[feat] > 0 else "−"
                significant_features.append((feat, direction, result.pvalues[feat]))

        if significant_features:
            lines.append("### Significant predictors of blunders (full model)")
            lines.append("")
            for feat, direction, pval in significant_features:
                lines.append(f"- **{feat}** ({direction}, p={pval:.4f})")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by chess-blunder-analysis pipeline*")

    findings_text = "\n".join(lines)
    findings_path = results_path / "findings.md"
    with open(findings_path, "w") as f:
        f.write(findings_text)
    print(f"Findings saved to {findings_path}")

    return findings_text


def run_analysis(csv_path="data/moves_features.csv", results_dir="results"):
    """Run the full analysis pipeline."""
    df = load_analysis_data(csv_path)

    if len(df) < 100:
        print("Not enough data for analysis. Need at least 100 observations.")
        return

    models = run_logistic_models(df, results_dir)
    run_hypothesis_tests(df, results_dir)
    generate_findings(df, models, results_dir)


if __name__ == "__main__":
    run_analysis()
