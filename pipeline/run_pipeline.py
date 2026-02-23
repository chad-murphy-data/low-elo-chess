#!/usr/bin/env python3
"""
Chess Blunder Hazard Model — Full Pipeline

Usage:
    python run_pipeline.py                  # Run all steps (with live refresh)
    python run_pipeline.py collect          # Step 1: Collect data from Lichess
    python run_pipeline.py features         # Step 2: Build feature dataset from PGNs
    python run_pipeline.py analyze          # Step 3: Run logistic regression + tests
    python run_pipeline.py plot             # Step 4: Generate all plots
    python run_pipeline.py dashboard        # Launch live dashboard on localhost:5050
    python run_pipeline.py --help           # Show this help

Steps can be run independently. Each step reads from / writes to the data/
and results/ directories.

The 'collect' step hits the Lichess API and can take a while depending on
how many users/games you target. The other steps are fast once data exists.

When running 'all' (default), features/analysis/plots refresh incrementally
every --refresh-every new PGN downloads so the dashboard stays up-to-date.
"""

import argparse
import sys
from pathlib import Path


def _make_refresh_callback(args):
    """Create a callback that runs features → analyze → plot."""
    def on_new_games(data_dir):
        from src.features import build_dataset
        from src.analysis import run_analysis
        from src.plots import generate_all_plots

        csv_path = Path(data_dir) / "moves_features.csv"
        fig_dir = Path(args.results_dir) / "figures"

        df = build_dataset(data_dir=data_dir)
        if len(df) >= 100:
            run_analysis(csv_path=str(csv_path), results_dir=args.results_dir)
            generate_all_plots(csv_path=str(csv_path), fig_dir=str(fig_dir))
        else:
            print(f"  Only {len(df)} observations -- skipping analysis (need >=100)")

    return on_new_games


def step_collect(args, live=False):
    """Step 1: Collect games from Lichess API via snowball sampling."""
    from src.collect import collect_data

    print("=" * 60)
    print("STEP 1: Collecting data from Lichess API")
    if live:
        print(f"  (live mode: refreshing analysis every {args.refresh_every} new downloads)")
    print("=" * 60)
    print()

    kwargs = dict(
        data_dir=args.data_dir,
        target_users=args.target_users,
        target_games=args.target_games,
        max_iterations=args.max_iterations,
    )
    if live:
        kwargs["on_new_games"] = _make_refresh_callback(args)
        kwargs["refresh_every"] = args.refresh_every

    collect_data(**kwargs)


def step_features(args):
    """Step 2: Parse PGN files into per-move feature dataset."""
    from src.features import build_dataset

    print("=" * 60)
    print("STEP 2: Building feature dataset from PGN files")
    print("=" * 60)
    print()

    df = build_dataset(data_dir=args.data_dir)
    print(f"\nDataset shape: {df.shape}")
    if len(df) > 0:
        print(f"Columns: {list(df.columns)}")
        print(f"\nELO distribution:")
        print(df["player_elo"].describe())


def step_analyze(args):
    """Step 3: Run logistic regression and hypothesis tests."""
    from src.analysis import run_analysis

    print("=" * 60)
    print("STEP 3: Running analysis")
    print("=" * 60)
    print()

    csv_path = Path(args.data_dir) / "moves_features.csv"
    run_analysis(csv_path=str(csv_path), results_dir=args.results_dir)


def step_plot(args):
    """Step 4: Generate all visualization plots."""
    from src.plots import generate_all_plots

    print("=" * 60)
    print("STEP 4: Generating plots")
    print("=" * 60)
    print()

    csv_path = Path(args.data_dir) / "moves_features.csv"
    fig_dir = Path(args.results_dir) / "figures"
    generate_all_plots(csv_path=str(csv_path), fig_dir=str(fig_dir))


def step_dashboard(args):
    """Launch live dashboard on localhost."""
    from src.dashboard import run_dashboard

    run_dashboard(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chess Blunder Hazard Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    valid_steps = {"all", "collect", "features", "analyze", "plot", "dashboard"}
    parser.add_argument(
        "steps", nargs="*", default=[],
        help="Which pipeline step(s) to run: all, collect, features, analyze, plot, dashboard (default: all)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Dashboard host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5050,
        help="Dashboard port (default: 5050)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory for data files (default: data/)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory for results (default: results/)",
    )
    parser.add_argument(
        "--target-users", type=int, default=200,
        help="Target users per ELO band for collection (default: 200)",
    )
    parser.add_argument(
        "--target-games", type=int, default=20,
        help="Games to download per user (default: 20)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Max snowball sampling iterations (default: 50)",
    )
    parser.add_argument(
        "--refresh-every", type=int, default=20,
        help="Refresh analysis every N new PGN downloads (default: 20)",
    )

    args = parser.parse_args()

    steps = args.steps if args.steps else ["all"]
    for s in steps:
        if s not in valid_steps:
            parser.error(f"invalid step: {s!r} (choose from {', '.join(sorted(valid_steps))})")
    if "all" in steps:
        # In 'all' mode, run collect with live refresh, then a final analysis pass
        step_collect(args, live=True)
        step_features(args)
        step_analyze(args)
        step_plot(args)
    else:
        step_funcs = {
            "collect": lambda a: step_collect(a, live=False),
            "features": step_features,
            "analyze": step_analyze,
            "plot": step_plot,
            "dashboard": step_dashboard,
        }
        for step_name in steps:
            step_funcs[step_name](args)
            print()

    print("=" * 60)
    print("Pipeline complete!")
    print(f"  Data: {args.data_dir}/")
    print(f"  Results: {args.results_dir}/")
    print(f"  Figures: {args.results_dir}/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
