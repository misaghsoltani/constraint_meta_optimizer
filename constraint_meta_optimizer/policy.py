from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
import torch.distributions as D


class GaussianPolicy(nn.Module):
    """Simple diagonal Gaussian policy network for continuous actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Store dimensions for visualization
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes

        # For tracking activations during forward passes
        self.last_activations = None
        self.activation_stats = {"mean_values": [], "std_values": []}

    def forward(self, obs: torch.Tensor) -> tuple[D.Normal, torch.Tensor]:
        """Forward pass through the policy network.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)

        Returns:
            dist: Normal distribution with mean and std
            h: Hidden features from the body network

        """
        h = self.body(obs)
        mean = self.mean(h)
        std = torch.exp(self.log_std)

        # Track distribution statistics for visualizations
        with torch.no_grad():
            self.activation_stats["mean_values"].append(mean.mean().item())
            self.activation_stats["std_values"].append(std.mean().item())
            if len(self.activation_stats["mean_values"]) > 1000:
                # Keep the history bounded to avoid memory issues
                self.activation_stats["mean_values"].pop(0)
                self.activation_stats["std_values"].pop(0)

        self.last_activations = h  # Store for visualization
        return D.Normal(mean, std), h  # return latent as well

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action from the policy given an observation.

        Args:
            obs: Observation tensor
            deterministic: If True, return the mean action instead of sampling

        Returns:
            Action tensor, clamped to [-1, 1]

        """
        dist, _ = self.forward(obs)
        return (dist.mean if deterministic else dist.sample()).clamp(-1.0, 1.0)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Calculate log probability of actions given observations.

        Args:
            obs: Observation tensor
            act: Action tensor

        Returns:
            Log probability tensor

        """
        dist, _ = self.forward(obs)
        return dist.log_prob(act).sum(-1, keepdim=True)

    def visualize_architecture(self) -> Figure:
        """Visualize the policy network architecture."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define layer positions
        layers = [self.obs_dim] + list(self.hidden_sizes) + [self.act_dim]
        n_layers = len(layers)
        layer_positions = np.linspace(0, 1, n_layers)
        max_neurons = max(layers)

        # Draw each layer
        for i, (pos, n_neurons) in enumerate(zip(layer_positions, layers, strict=False)):
            # Adjust vertical positions for neurons in this layer
            spacing = 0.8 / max(n_neurons, 1)
            start = 0.1 + spacing / 2
            neuron_ys = np.linspace(start, 0.9 - spacing / 2, n_neurons)

            # Draw neurons
            for y in neuron_ys:
                circle = plt.Circle((pos, y), 0.01, color="b", fill=True)
                ax.add_patch(circle)

            # Draw connections to next layer if not the last layer
            if i < n_layers - 1:
                next_n_neurons = layers[i + 1]
                next_spacing = 0.8 / max(next_n_neurons, 1)
                next_start = 0.1 + next_spacing / 2
                next_ys = np.linspace(next_start, 0.9 - next_spacing / 2, next_n_neurons)

                for y1 in neuron_ys:
                    for y2 in next_ys:
                        ax.plot([pos, layer_positions[i + 1]], [y1, y2], "k-", alpha=0.1)

        # Add layer labels
        for i, (pos, n_neurons) in enumerate(zip(layer_positions, layers, strict=False)):
            if i == 0:
                label = f"Input\n{n_neurons} units"
            elif i == n_layers - 1:
                label = f"Output\n{n_neurons} units"
            else:
                label = f"Hidden {i}\n{n_neurons} units"
            ax.text(pos, 0.02, label, ha="center")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.set_title("Policy Network Architecture")
        ax.axis("off")

        return fig

    def visualize_action_distribution(self, observations: torch.Tensor, title: str | None = None) -> Figure:
        """Visualize the action distribution for a batch of observations."""
        with torch.no_grad():
            dist, _ = self.forward(observations)
            means = dist.mean.detach().cpu().numpy()
            stds = dist.scale.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot action means
        ax = axes[0]
        if self.act_dim <= 10:  # For low-dimensional action spaces
            for dim in range(self.act_dim):
                sns.kdeplot(means[:, dim], ax=ax, label=f"Dim {dim}")
            ax.legend()
        else:  # For high-dimensional action spaces
            sns.heatmap(means.T, ax=ax, cmap="viridis")
            ax.set_xlabel("Batch Item")
            ax.set_ylabel("Action Dimension")
        ax.set_title("Action Mean Distribution")

        # Plot action standard deviations
        ax = axes[1]
        if self.act_dim <= 10:  # For low-dimensional action spaces
            for dim in range(self.act_dim):
                sns.kdeplot(stds[:, dim], ax=ax, label=f"Dim {dim}")
            ax.legend()
        else:  # For high-dimensional action spaces
            sns.heatmap(stds.T, ax=ax, cmap="viridis")
            ax.set_xlabel("Batch Item")
            ax.set_ylabel("Action Dimension")
        ax.set_title("Action Std Distribution")

        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Policy Action Distribution (batch size: {observations.shape[0]})", fontsize=16)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def visualize_distribution_over_time(self) -> Figure:
        """Visualize how the policy distribution has changed over time."""
        if not self.activation_stats["mean_values"]:
            # No data yet
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No distribution history available yet", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot mean and std evolution
        steps = np.arange(len(self.activation_stats["mean_values"]))
        ax.plot(steps, self.activation_stats["mean_values"], "b-", label="Mean Value")
        ax.plot(steps, self.activation_stats["std_values"], "r-", label="Std Value")

        # Rolling average for smoothing
        window_size = min(50, len(steps) // 10) if len(steps) > 100 else 1
        if window_size > 1:
            mean_smooth = np.convolve(
                self.activation_stats["mean_values"], np.ones(window_size) / window_size, mode="valid"
            )
            std_smooth = np.convolve(
                self.activation_stats["std_values"], np.ones(window_size) / window_size, mode="valid"
            )
            smooth_steps = np.arange(window_size - 1, len(steps))

            ax.plot(smooth_steps, mean_smooth, "b-", linewidth=2, alpha=0.8, label="Mean (Smoothed)")
            ax.plot(smooth_steps, std_smooth, "r-", linewidth=2, alpha=0.8, label="Std (Smoothed)")

        ax.set_xlabel("Policy Update Steps")
        ax.set_ylabel("Value")
        ax.set_title("Policy Distribution Parameters Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add information about current values
        current_mean = self.activation_stats["mean_values"][-1] if self.activation_stats["mean_values"] else 0
        current_std = self.activation_stats["std_values"][-1] if self.activation_stats["std_values"] else 0
        ax.text(
            0.02,
            0.95,
            f"Current Mean: {current_mean:.4f}\nCurrent Std: {current_std:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        return fig
