"""
market_impact/run_analysis.py
─────────────────────────────
Entry point: runs the full market impact pipeline end-to-end.

Usage
-----
    python market_impact/run_analysis.py          # full pipeline
    python market_impact/run_analysis.py --plot   # plot only (skip compute)
"""

import os
import sys
import argparse
import time

# Ensure the project root and current directory are on the path
sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Market Impact Pipeline")
    parser.add_argument("--plot", action="store_true",
                        help="Skip data computation and go straight to plotting.")
    args = parser.parse_args()

    t0 = time.time()

    if not args.plot:
        print("=" * 60)
        print("  Step 1/5 — Computing market impact metrics")
        print("=" * 60)
        from compute_impact import main as compute_main
        compute_main()
        elapsed = time.time() - t0
        print(f"\n  ✓ Compute step done in {elapsed/60:.1f} min\n")

    print("=" * 60)
    print("  Step 2/5 — Generating publication-quality figures")
    print("=" * 60)
    from plot_impact import main as plot_main
    plot_main()
    
    print("\n  Step 3/5 — Generating Zoom-in Microstructure Plots (All Cases)")
    print("=" * 60)
    from plot_zoom_all import main as plot_zoom_all_main
    plot_zoom_all_main()
    
    print("\n  Step 4/5 — Generating Figure 8: Trade Size Relationship Plot")
    print("=" * 60)
    from plot_size_relationship import generate_relationship_plot
    generate_relationship_plot()

    print("\n  Step 5/5 — Generating Filtered Market Impact Plots (Trade Size >= $100)")
    print("=" * 60)
    from plot_impact_filtered import main as plot_filtered_main
    plot_filtered_main()

    total = time.time() - t0
    print(f"\n  [Done] Full pipeline complete in {total/60:.1f} min")
    print(f"  Figures -> plots/")


if __name__ == "__main__":
    main()
