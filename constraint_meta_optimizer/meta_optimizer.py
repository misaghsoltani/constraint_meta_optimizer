import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class PIDLagrangianOptimizer(nn.Module):
    """PID Lagrangian meta-optimizer as described in the project submission.

    Implements the structured form:
    - Policy update (primal): Δθ_k = η_R * ∇_θ J_R - λ_k * η_C * ∇_θ J_C
    - Dual update (PID): λ_k+1 = max{0, λ_k + K_P*v_k + K_I*Σv_i + K_D*(v_k - v_k-1)}

    where v_k = J_C(π_θ_k) - β is the constraint violation.
    """

    def __init__(
        self,
        cost_limit: float = 25.0,
        init_KP: float = 0.05,
        init_KI: float = 0.01,
        init_KD: float = 0.005,
        init_etaR: float = 3e-4,
        init_etaC: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__()

        # Meta-parameters (learnable)
        self.K_P = nn.Parameter(torch.tensor([init_KP], device=device))
        self.K_I = nn.Parameter(torch.tensor([init_KI], device=device))
        self.K_D = nn.Parameter(torch.tensor([init_KD], device=device))
        self.eta_R = nn.Parameter(torch.tensor([init_etaR], device=device))
        self.eta_C = nn.Parameter(torch.tensor([init_etaC], device=device))

        # Fixed parameters
        self.cost_limit = cost_limit
        self.device = device

        # State tracking for PID
        self.prev_violations = []  # Store violation history for integral term
        self.last_violation = 0.0  # Store previous violation for derivative term

        # For visualizing the PID gains over time
        self.KP_history = []
        self.KI_history = []
        self.KD_history = []
        self.etaR_history = []
        self.etaC_history = []

    def forward(
        self,
        reward_grad: torch.Tensor,
        cost_grad: torch.Tensor,
        lambda_val: torch.Tensor,
        constraint_val: torch.Tensor,
        prev_constraint_val: torch.Tensor = None,
    ):
        """Compute policy and lambda updates according to PID Lagrangian.

        Args:
            reward_grad: Policy gradient w.r.t reward (∇_θ J_R)
            cost_grad: Policy gradient w.r.t cost (∇_θ J_C)
            lambda_val: Current Lagrange multiplier value
            constraint_val: Current constraint violation (J_C - β)
            prev_constraint_val: Previous constraint violation

        Returns:
            delta_theta: Policy parameter update
            delta_lambda: Lagrange multiplier update

        """
        # Track history for visualization
        self.KP_history.append(self.K_P.item())
        self.KI_history.append(self.K_I.item())
        self.KD_history.append(self.K_D.item())
        self.etaR_history.append(self.eta_R.item())
        self.etaC_history.append(self.eta_C.item())

        # Update violation history for integral term
        self.prev_violations.append(constraint_val.item())
        if len(self.prev_violations) > 1000:  # Limit history to avoid memory issues
            self.prev_violations.pop(0)

        # Compute integral term
        integral_term = sum(self.prev_violations)

        # Compute derivative term
        if prev_constraint_val is not None:
            derivative_term = constraint_val - prev_constraint_val
        else:
            derivative_term = torch.zeros_like(constraint_val)

        # Update last_violation for next iteration
        self.last_violation = constraint_val.item()

        # Policy update (primal)
        delta_theta = self.eta_R * reward_grad - lambda_val * self.eta_C * cost_grad

        # Lambda update (dual) - PID form
        delta_lambda = self.K_P * constraint_val + self.K_I * integral_term + self.K_D * derivative_term

        return delta_theta, delta_lambda

    def visualize_pid_gains(self):
        """Visualize the evolution of PID gains during meta-training."""
        if not self.KP_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No PID gain history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot PID gains
        ax = axes[0]
        steps = np.arange(len(self.KP_history))
        ax.plot(steps, self.KP_history, "r-", label="K_P (Proportional)")
        ax.plot(steps, self.KI_history, "g-", label="K_I (Integral)")
        ax.plot(steps, self.KD_history, "b-", label="K_D (Derivative)")
        ax.set_ylabel("PID Gain Value")
        ax.set_title("Evolution of PID Gains")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot learning rate scalars
        ax = axes[1]
        ax.plot(steps, self.etaR_history, "c-", label="η_R (Reward)")
        ax.plot(steps, self.etaC_history, "m-", label="η_C (Cost)")
        ax.set_xlabel("Meta-Training Iteration")
        ax.set_ylabel("Learning Rate Value")
        ax.set_title("Evolution of Learning Rate Scalars")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def get_meta_parameters(self) -> dict[str, float]:
        """Return current meta-parameter values."""
        return {
            "K_P": self.K_P.item(),
            "K_I": self.K_I.item(),
            "K_D": self.K_D.item(),
            "eta_R": self.eta_R.item(),
            "eta_C": self.eta_C.item(),
        }

    def reset_hidden_state(self) -> None:
        """Reset PID optimizer state between tasks."""
        # Clear stored violations and integrals
        self.prev_violations = []
        self.last_violation = 0.0
        # Clear PID gain and learning rate history
        self.KP_history = []
        self.KI_history = []
        self.KD_history = []
        self.etaR_history = []
        self.etaC_history = []


class MetaOptimizerRNN(nn.Module):
    """RNN-based meta-optimizer that outputs \Delta\theta and \Delta\lambda.

    The network sees per-parameter gradients concatenated with the current
    constraint signal and Lagrange multiplier. To keep dimensionality
    manageable, gradients are first aggregated with layer-norm and average
    pooling.
    """

    def __init__(
        self,
        grad_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        init_KP: float = 0.05,
        init_KI: float = 0.01,
        init_KD: float = 0.005,
        init_etaR: float = 3e-4,
        init_etaC: float = 3e-4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=grad_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.delta_head = nn.Linear(hidden_size, grad_dim)  # outputs \Delta params (+ multiplier)
        nn.init.zeros_(self.delta_head.bias)

        # PID controller learnable parameters
        self.K_P = nn.Parameter(torch.tensor([init_KP]))
        self.K_I = nn.Parameter(torch.tensor([init_KI]))
        self.K_D = nn.Parameter(torch.tensor([init_KD]))
        self.eta_R = nn.Parameter(torch.tensor([init_etaR]))
        self.eta_C = nn.Parameter(torch.tensor([init_etaC]))

        # For tracking hidden state and update statistics for visualization
        self.hidden_histories = []
        self.update_magnitudes = []
        self.lambda_updates = []
        self.theta_updates = []
        self.grad_norms = []

        # For tracking PID values
        self.pid_values = {"kp": [], "ki": [], "kd": []}
        self.violation_integral = 0.0
        self.prev_violation = 0.0

        # Track hidden state evolution for a specific sample to visualize how the LSTM evolves
        self.reference_hiddens = None
        self.max_history_length = 1000  # Limit history size to avoid memory issues

    def get_pid_parameters(self):
        """Return the current PID gain parameters."""
        return {
            "K_P": self.K_P.item(),
            "K_I": self.K_I.item(),
            "K_D": self.K_D.item(),
            "eta_R": self.eta_R.item(),
            "eta_C": self.eta_C.item(),
        }

    def F_phi(self, grad_R, grad_C, lambda_value):
        """Policy (primal) update function F_φ according to paper.

        Args:
            grad_R: gradient of reward objective
            grad_C: gradient of cost objective
            lambda_value: current Lagrange multiplier value

        Returns:
            Delta_theta: Policy parameter update

        """
        return self.eta_R * grad_R - lambda_value * self.eta_C * grad_C

    def G_phi(self, violation, prev_violation, integral):
        """Dual (lambda) update function G_φ according to paper.

        Args:
            violation: current constraint violation (J_C - beta)
            prev_violation: previous constraint violation
            integral: sum of all constraint violations

        Returns:
            Delta_lambda: Lagrange multiplier update

        """
        return self.K_P * violation + self.K_I * integral + self.K_D * (violation - prev_violation)

    def forward(
        self, grad_seq: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Args:
            grad_seq: (B, T, D) gradient slices over unroll window
            hx: Optional tuple of (hidden state, cell state) for LSTM

        Returns:
            deltas: (B, T, D) parameter updates
            hidden: Tuple of (hidden state, cell state)

        """
        # Remove all .item() and Python-side logging from the forward path for Dynamo/JIT compatibility
        # If diagnostics are needed, call a separate method after forward

        if (
            hx is None
            and hasattr(self, "hidden")
            and hasattr(self, "cell")
            and self.hidden is not None
            and self.cell is not None
        ):
            hx = (self.hidden, self.cell)

        out, new_hx = self.lstm(grad_seq, hx)  # (B, T, hidden)
        if isinstance(new_hx, tuple):
            self.hidden, self.cell = new_hx

        deltas = self.delta_head(out)
        return deltas, new_hx

    def log_diagnostics(
        self, grad_seq: torch.Tensor, deltas: torch.Tensor, new_hx: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Call this outside of forward for diagnostics and history tracking."""
        with torch.no_grad():
            # Make sure all inputs are properly detached
            grad_seq = grad_seq.detach() if grad_seq.requires_grad else grad_seq
            deltas = deltas.detach() if deltas.requires_grad else deltas

            # Process batch data
            batch_size, seq_len, dim = grad_seq.shape
            flat_grad = grad_seq.view(batch_size * seq_len, -1)
            grad_norm = torch.norm(flat_grad, dim=1).mean().item()
            self.grad_norms.append(grad_norm)
            if len(self.grad_norms) > self.max_history_length:
                self.grad_norms.pop(0)

            # Track PID values for visualization
            self.pid_values["kp"].append(self.K_P.item())
            self.pid_values["ki"].append(self.K_I.item())
            self.pid_values["kd"].append(self.K_D.item())
            if len(self.pid_values["kp"]) > self.max_history_length:
                self.pid_values["kp"].pop(0)
                self.pid_values["ki"].pop(0)
                self.pid_values["kd"].pop(0)

            # Store hidden state evolution if it's the first sample or every 100 steps
            if (
                (len(self.hidden_histories) % 100 == 0 or not self.hidden_histories)
                and isinstance(new_hx, tuple)
                and new_hx[0] is not None
            ):
                h_n = new_hx[0][:, 0, :].detach().cpu().numpy()
                self.hidden_histories.append(h_n)
                if len(self.hidden_histories) > self.max_history_length:
                    self.hidden_histories.pop(0)

            # Track update magnitudes
            update_mag = torch.norm(deltas, dim=2).mean().item()
            self.update_magnitudes.append(update_mag)
            if len(self.update_magnitudes) > self.max_history_length:
                self.update_magnitudes.pop(0)

            # Track theta and lambda updates separately
            if deltas.size(2) > 2:
                theta_update = deltas[0, 0, :-2]
                lambda_update = deltas[0, 0, -2]
                self.theta_updates.append(torch.norm(theta_update).item())
                self.lambda_updates.append(lambda_update.item())
                if len(self.theta_updates) > self.max_history_length:
                    self.theta_updates.pop(0)
                if len(self.lambda_updates) > self.max_history_length:
                    self.lambda_updates.pop(0)

    def visualize_pid_evolution(self):
        """Visualize how PID gains evolve during meta-training."""
        if not self.pid_values["kp"]:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No PID history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.pid_values["kp"]))

        ax.plot(x, self.pid_values["kp"], "r-", label="K_P (Proportional)")
        ax.plot(x, self.pid_values["ki"], "g-", label="K_I (Integral)")
        ax.plot(x, self.pid_values["kd"], "b-", label="K_D (Derivative)")

        ax.set_xlabel("Meta-Update Step")
        ax.set_ylabel("Gain Value")
        ax.set_title("PID Controller Gain Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add current values in text box
        stats_text = (
            f"Current Values:\n"
            f"K_P: {self.K_P.item():.4f}\n"
            f"K_I: {self.K_I.item():.4f}\n"
            f"K_D: {self.K_D.item():.4f}\n"
            f"η_R: {self.eta_R.item():.4f}\n"
            f"η_C: {self.eta_C.item():.4f}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

        return fig

    def visualize_hidden_evolution(self):
        """Visualize how hidden states evolve during optimization."""
        if not self.hidden_histories:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No hidden state history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        # Select a subset of hidden dimensions to visualize (first 10)
        vis_dims = min(10, self.hidden_histories[0].shape[1])

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.hidden_histories))

        for i in range(vis_dims):
            hiddens = [h[0, i] for h in self.hidden_histories]  # First layer, i-th dimension
            ax.plot(x, hiddens, label=f"Dim {i}", alpha=0.7)

        ax.set_xlabel("Update Step (sampled every 100 steps)")
        ax.set_ylabel("Hidden State Value")
        ax.set_title("LSTM Hidden State Evolution")

        # Only show legend if not too many dimensions
        if vis_dims <= 5:
            ax.legend()

        ax.grid(True, alpha=0.3)
        return fig

    def visualize_update_magnitudes(self):
        """Visualize the magnitude of parameter updates over time."""
        if not self.update_magnitudes:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No update history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.update_magnitudes))

        # Plot raw magnitudes
        ax.plot(x, self.update_magnitudes, "b-", alpha=0.4, label="Update Magnitude")

        # Add smoothed line
        window_size = min(50, max(1, len(x) // 20))
        if window_size > 1:
            smoothed = np.convolve(self.update_magnitudes, np.ones(window_size) / window_size, mode="valid")
            smoothed_x = np.arange(window_size - 1, len(x))
            ax.plot(smoothed_x, smoothed, "r-", linewidth=2, label=f"Smoothed (window={window_size})")

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Update Magnitude (L2 Norm)")
        ax.set_title("Parameter Update Magnitudes Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics in text box
        mean_mag = np.mean(self.update_magnitudes)
        max_mag = np.max(self.update_magnitudes)
        recent_mean = np.mean(self.update_magnitudes[-min(100, len(self.update_magnitudes)) :])

        stats_text = f"Mean: {mean_mag:.4f}\nMax: {max_mag:.4f}\nRecent Mean: {recent_mean:.4f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

        return fig

    def visualize_theta_lambda_updates(self):
        """Visualize policy parameter vs lambda parameter updates."""
        if not self.theta_updates or not self.lambda_updates:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No update history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.theta_updates))

        # Plot on different scales for comparison
        ax.plot(x, self.theta_updates, "b-", alpha=0.7, label="Policy Updates (θ)")

        # Add second y-axis for lambda
        ax2 = ax.twinx()
        ax2.plot(x, self.lambda_updates, "r-", alpha=0.7, label="Lambda Updates (λ)")

        # Window size for smoothing
        window_size = min(50, max(1, len(x) // 20))
        if window_size > 1:
            # Smooth theta updates
            smoothed_theta = np.convolve(self.theta_updates, np.ones(window_size) / window_size, mode="valid")
            smoothed_x = np.arange(window_size - 1, len(x))
            ax.plot(smoothed_x, smoothed_theta, "b-", linewidth=2)

            # Smooth lambda updates
            smoothed_lambda = np.convolve(self.lambda_updates, np.ones(window_size) / window_size, mode="valid")
            ax2.plot(smoothed_x, smoothed_lambda, "r-", linewidth=2)

        # Labels and title
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Policy Update Magnitude (θ)", color="b")
        ax.tick_params(axis="y", labelcolor="b")

        ax2.set_ylabel("Lambda Update Value (λ)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        fig.suptitle("Policy (θ) vs Constraint (λ) Parameter Updates", fontsize=14)

        # Add stats in box
        theta_mean = np.mean(self.theta_updates)
        lambda_mean = np.mean(self.lambda_updates)
        lambda_std = np.std(self.lambda_updates)

        stats_text = f"θ Mean: {theta_mean:.4f}\nλ Mean: {lambda_mean:.4f}\nλ Std: {lambda_std:.4f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def visualize_grad_norms(self):
        """Visualize the evolution of gradient norms over time."""
        if not self.grad_norms:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No gradient norm history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.grad_norms))

        # Plot raw gradient norms
        ax.plot(x, self.grad_norms, "g-", alpha=0.4, label="Gradient Norm")

        # Add smoothed line
        window_size = min(50, max(1, len(x) // 20))
        if window_size > 1:
            smoothed = np.convolve(self.grad_norms, np.ones(window_size) / window_size, mode="valid")
            smoothed_x = np.arange(window_size - 1, len(x))
            ax.plot(smoothed_x, smoothed, "b-", linewidth=2, label=f"Smoothed (window={window_size})")

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Policy Gradient Norms Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics in text box
        if self.grad_norms:
            mean_norm = np.mean(self.grad_norms)
            max_norm = np.max(self.grad_norms)
            recent_mean = np.mean(self.grad_norms[-min(100, len(self.grad_norms)) :])

            stats_text = f"Mean: {mean_norm:.4f}\nMax: {max_norm:.4f}\nRecent Mean: {recent_mean:.4f}"
            ax.text(
                0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.8)
            )

        return fig

    def reset_visualization_history(self):
        """Reset all tracking variables for visualizations."""
        self.hidden_histories = []
        self.update_magnitudes = []
        self.lambda_updates = []
        self.theta_updates = []
        self.grad_norms = []
        self.reference_hiddens = None
        self.pid_values = {"kp": [], "ki": [], "kd": []}  # Reset PID history

    def reset_hidden_state(self) -> None:
        """Reset the LSTM hidden state to ensure clean gradient flow between tasks."""
        # The actual hidden state is created on-the-fly during forward pass
        # This method primarily exists to signal intent to clear computational graph
        self.reference_hiddens = None
        # Initialize hidden state attributes if needed
        self.hidden = None
        self.cell = None
        # Reset integral term for PID controller to avoid accumulating across tasks
        self.violation_integral = 0.0
        self.prev_violation = 0.0
