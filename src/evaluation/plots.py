"""Precision/Latency tradeoff visualization module for Day 4 evaluation.

Key Invariants:
- STRICT DEVELOPMENT DATA ONLY: strictly prohibits access to holdout data paths ('data/holdout/').
- Visualizes precision vs latency trade-off across development operating points.
- Uses headless Agg backend for matplotlib.
"""

from typing import List, Dict, Any, Optional, Sequence, Union
from pathlib import Path

from src.evaluation.sweeps import _verify_development_only_data, HoldoutAccessViolationError


def generate_precision_latency_tradeoff_plot(
    sweep_results: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    data_path: Optional[str] = None,
    title: str = "Precision vs Latency Trade-off Across Operating Points",
) -> Path:
    """Generate precision vs latency tradeoff plot from development parameter sweep results.

    Args:
        sweep_results: List of dictionary records containing precision, median_latency_seconds, p95_latency_seconds, etc.
        output_path: Target path to save the generated plot image.
        data_path: Path of the input dataset (checked against holdout firewall).
        title: Plot title.

    Returns:
        Path to the saved plot image.
    """
    _verify_development_only_data(data_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Filter records with non-null precision and latency
    valid_records = [
        r for r in sweep_results
        if r.get("precision") is not None and r.get("median_latency_seconds") is not None
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if valid_records:
        precisions = [r["precision"] for r in valid_records]
        latencies = [r["median_latency_seconds"] for r in valid_records]
        labels = [
            f"th={r.get('threshold', r.get('alpha', ''))}"
            for r in valid_records
        ]

        # Scatter / Line plot of Precision vs Median Latency
        scatter = ax1.scatter(latencies, precisions, color="royalblue", s=80, zorder=5, label="Operating Points")
        for i, txt in enumerate(labels):
            if txt and txt != "th=":
                ax1.annotate(txt, (latencies[i], precisions[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Plot Pareto curve if multiple points
        sorted_indices = sorted(range(len(latencies)), key=lambda i: latencies[i])
        sorted_lat = [latencies[i] for i in sorted_indices]
        sorted_prec = [precisions[i] for i in sorted_indices]
        ax1.plot(sorted_lat, sorted_prec, linestyle="--", color="gray", alpha=0.7, label="Trade-off Curve")

    ax1.set_xlabel("Median Detection Latency (seconds)")
    ax1.set_ylabel("Precision")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(title)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close(fig)

    return out_p
