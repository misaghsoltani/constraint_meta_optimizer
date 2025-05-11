from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


class ResearchLogger:
    """Enhanced logger for research experiments with multiple visualization options."""

    def __init__(
        self,
        logdir: str,
        experiment_name: str | None = None,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a comprehensive research logger.

        Args:
            logdir: Base directory for logs
            experiment_name: Name of the experiment (defaults to timestamp)
            use_tensorboard: Whether to use TensorBoard logging
            use_csv: Whether to use CSV logging
            metadata: Dictionary of experiment metadata

        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.experiment_name = experiment_name
        self.base_logdir = Path(logdir)
        self.logdir = self.base_logdir / experiment_name
        os.makedirs(self.logdir, exist_ok=True)

        # TensorBoard logging
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.logdir))

        # CSV logging
        self.use_csv = use_csv
        if use_csv:
            self.csv_path = self.logdir / "metrics.csv"
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = None

        # Statistics tracking
        self.buffer = defaultdict(list)
        self.episode_data = []
        self.steps = defaultdict(int)

        # Save experiment metadata
        if metadata:
            with open(self.logdir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        # Set up plot directory
        self.plot_dir = self.logdir / "plots"
        os.makedirs(self.plot_dir, exist_ok=True)

        self.start_time = time.time()
        print(f"📊 Logging to {self.logdir}")

    def log(self, step: int | None = None, **metrics) -> None:
        """Log metrics for the current step."""
        for k, v in metrics.items():
            self.buffer[k].append(float(v) if isinstance(v, (int, float, np.number)) else v)

            # Log to TensorBoard directly if numerical
            if self.use_tensorboard and isinstance(v, (int, float, np.number)):
                step_key = step if step is not None else self.steps["default"]
                self.writer.add_scalar(k, v, step_key)

        if step is None:
            self.steps["default"] += 1

    def log_histogram(self, key: str, values: np.ndarray | list, step: int | None = None) -> None:
        """Log a histogram of values in TensorBoard."""
        if self.use_tensorboard:
            step_key = step if step is not None else self.steps["default"]
            self.writer.add_histogram(key, np.array(values), step_key)

    def log_image(self, key: str, figure: Figure, step: int | None = None) -> None:
        """Log a matplotlib figure in TensorBoard and save to disk."""
        # Save to TensorBoard
        if self.use_tensorboard:
            step_key = step if step is not None else self.steps["default"]
            self.writer.add_figure(key, figure, step_key)

        # Save to disk
        output_path = self.plot_dir / f"{key}_{step if step is not None else 'latest'}.png"
        figure.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(figure)

    def log_episode(self, episode_metrics: dict[str, Any]) -> None:
        """Log complete episode data including trajectory information."""
        self.episode_data.append(episode_metrics)

        # Extract main metrics for summary statistics
        if "episode_return" in episode_metrics:
            self.log(episode_return=episode_metrics["episode_return"])
        if "episode_constraint_violation" in episode_metrics:
            self.log(episode_constraint_violation=episode_metrics["episode_constraint_violation"])
        if "episode_length" in episode_metrics:
            self.log(episode_length=episode_metrics["episode_length"])

    def flush(self) -> dict[str, float]:
        """Compute statistics for the current buffer and clear it."""
        if not self.buffer:
            return {}

        # Compute statistics for numerical values
        stats = {}
        for k, v in self.buffer.items():
            if all(isinstance(x, (int, float, np.number)) for x in v):
                v_array = np.array(v)
                stats[f"{k}/mean"] = v_array.mean()
                stats[f"{k}/std"] = v_array.std()
                stats[f"{k}/min"] = v_array.min()
                stats[f"{k}/max"] = v_array.max()
                stats[f"{k}/median"] = np.median(v_array)
            elif len(v) == 1:
                stats[k] = v[0]  # Just use the value directly if only one

        # Log to CSV
        if self.use_csv:
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=stats.keys())
                self.csv_writer.writeheader()
            self.csv_writer.writerow(stats)
            self.csv_file.flush()

        # Clear buffer
        self.buffer.clear()
        return stats

    def create_training_curves(self, columns: list[str] | None = None, window_size: int = 10) -> Figure:
        """Create publication-quality training curves from logged data."""
        # If CSV exists, load it as a DataFrame
        if self.use_csv and os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)

            # If no columns specified, use numerical columns
            if columns is None:
                columns = [col for col in df.columns if df[col].dtype in (np.float64, np.int64)]

            # Create a figure with subplots
            n_cols = min(2, len(columns))
            n_rows = (len(columns) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)

            # Plot each metric with rolling mean
            for i, col in enumerate(columns):
                row, col_idx = i // 2, i % 2
                if col in df.columns:
                    # Apply rolling window
                    df[f"{col}_rolling"] = df[col].rolling(window=window_size).mean()

                    # Plot raw data as light scatter and rolling mean as solid line
                    ax = axes[row, col_idx]
                    ax.scatter(df.index, df[col], alpha=0.3, s=10, label=f"{col} (raw)")
                    ax.plot(df.index, df[f"{col}_rolling"], linewidth=2, label=f"{col} (avg{window_size})")
                    ax.set_title(col)
                    ax.set_xlabel("Step")
                    ax.legend()

                    # Add y-axis label only for leftmost plots
                    if col_idx == 0:
                        ax.set_ylabel("Value")

            fig.tight_layout()
            return fig

        return plt.figure()  # Return empty figure if no data

    def plot_training_progress(self, step: int, metrics: dict[str, float]) -> None:
        """Create and log a training progress summary plot."""
        # Create a training curve plot
        fig = self.create_training_curves(window_size=10)
        self.log_image("training_curves", fig, step)

        # Elapsed time and ETA
        elapsed_time = time.time() - self.start_time
        print(
            f"📈 Step {step}: Elapsed time: {elapsed_time:.1f}s | "
            f"Latest metrics: "
            + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        )

    def create_constraint_violation_plot(self, cost_data: list[float], cost_limit: float, step: int) -> Figure:
        """Create a constraint violation visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot cost over episodes
        episodes = np.arange(1, len(cost_data) + 1)
        ax.plot(episodes, cost_data, "o-", label="Episode Cost", alpha=0.7)

        # Plot cost limit as a horizontal line
        ax.axhline(y=cost_limit, color="r", linestyle="--", label=f"Cost Limit ({cost_limit})")

        # Calculate and show violation percentage
        violations = sum(1 for c in cost_data if c > cost_limit)
        violation_pct = (violations / len(cost_data)) * 100 if cost_data else 0
        ax.text(
            0.02,
            0.95,
            f"Violation Rate: {violation_pct:.1f}%",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Formatting
        ax.set_title(f"Constraint Violations (Step {step})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def close(self) -> None:
        """Close all logging resources."""
        if self.use_tensorboard:
            self.writer.close()
        if self.use_csv:
            self.csv_file.close()


# For backward compatibility
class CSVLogger:
    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        self.file = open(os.path.join(logdir, "metrics.csv"), "w", newline="")
        self.writer = None
        self.buffer = defaultdict(list)

    def log(self, **metrics):
        for k, v in metrics.items():
            self.buffer[k].append(v)

    def flush(self):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=self.buffer.keys())
            self.writer.writeheader()
        row = {k: sum(v) / len(v) for k, v in self.buffer.items()}
        self.writer.writerow(row)
        self.file.flush()
        self.buffer.clear()
