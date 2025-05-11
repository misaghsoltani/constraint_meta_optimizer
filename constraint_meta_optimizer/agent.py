from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from .meta_optimizer import MetaOptimizerRNN, PIDLagrangianOptimizer
from .policy import GaussianPolicy


@dataclass
class Trajectory:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    log_probs: torch.Tensor

    def get_summary_metrics(self) -> dict[str, float]:
        """Return summary statistics of this trajectory."""
        with torch.no_grad():
            metrics = {
                "return": self.rewards.sum().item(),
                "cost": self.costs.sum().item(),
                "trajectory_length": len(self.rewards),
                "mean_reward": self.rewards.mean().item(),
                "mean_cost": self.costs.mean().item(),
                "std_reward": self.rewards.std().item(),
                "std_cost": self.costs.std().item(),
                "max_reward": self.rewards.max().item(),
                "max_cost": self.costs.max().item(),
            }
        return metrics

    def visualize(self, step: int | None = None) -> plt.Figure:
        """Create a visualization of rewards and costs over time."""
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Get data as numpy arrays
        t = np.arange(len(self.rewards))
        rewards = self.rewards.cpu().numpy()
        costs = self.costs.cpu().numpy()

        # Plot rewards
        ax1.plot(t, rewards, "b-", label="Rewards", marker="o", markersize=3)
        ax1.axhline(y=rewards.mean(), color="r", linestyle="--", label=f"Mean: {rewards.mean():.3f}")
        ax1.fill_between(t, rewards, alpha=0.2)
        ax1.set_ylabel("Reward")
        ax1.set_title("Trajectory Rewards Over Time")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot costs
        ax2.plot(t, costs, "g-", label="Costs", marker="o", markersize=3)
        ax2.axhline(y=costs.mean(), color="r", linestyle="--", label=f"Mean: {costs.mean():.3f}")
        ax2.fill_between(t, costs, alpha=0.2)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Cost")
        ax2.set_title("Safety Costs Over Time")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Annotations
        title = f"Trajectory Analysis - Step {step}" if step is not None else "Trajectory Analysis"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout, leaving space for suptitle

        return fig


class PrimalDualAgent:
    """Constrained policy gradient agent with optional meta-optimizer."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cost_limit: float,
        meta_optimizer: MetaOptimizerRNN | PIDLagrangianOptimizer | None = None,
        alpha_theta: float = 3e-4,
        alpha_lambda: float = 1e-2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = GaussianPolicy(obs_dim, act_dim).to(self.device)
        self.lambda_param = torch.nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)

        self.meta_optimizer = meta_optimizer
        self.is_pid_optimizer = isinstance(meta_optimizer, PIDLagrangianOptimizer)
        if self.meta_optimizer is None:
            self.opt = torch.optim.Adam(self.policy.parameters(), lr=alpha_theta)
        self.alpha_lambda = alpha_lambda
        self.cost_limit = cost_limit

        # Tracking statistics
        self.lambda_history = []
        self.constraint_history = []
        self.policy_loss_history = []
        self.episode_count = 0
        self.update_count = 0

        print(f"🤖 PrimalDualAgent initialized with cost limit: {cost_limit}")
        print(f"   - Policy: {obs_dim}→{act_dim} with {sum(p.numel() for p in self.policy.parameters())} parameters")
        print(f"   - Lambda: {self.lambda_param.item():.4f} (initial value)")
        print(f"   - {'Meta-optimizer' if meta_optimizer else 'Standard optimizer'} mode\n")

    def select_action(self, obs, deterministic=False, return_log_prob=False):
        """Select action from policy with optional deterministic mode.

        Args:
            obs: Environment observation
            deterministic: If True, use mean action instead of sampling
            return_log_prob: If True, also return log probability of the action

        Returns:
            action: Selected action
            log_prob: Log probability of the action (optional)

        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Only use torch.no_grad() for evaluation (deterministic and not returning log_prob)
        if deterministic and not return_log_prob:
            with torch.no_grad():
                action_dist, _ = self.policy.forward(obs_t)
                act = action_dist.mean
                act_np = act.cpu().numpy()[0]
                return act_np
        else:
            action_dist, _ = self.policy.forward(obs_t)
            act = action_dist.mean if deterministic else action_dist.sample()
            act_np = act.cpu().numpy()[0]
            if return_log_prob:
                log_prob = action_dist.log_prob(act).sum(dim=-1).detach().cpu().numpy()[0]
                return act_np, log_prob
            return act_np

    def compute_loss(self, traj: Trajectory, gamma: float = 0.99) -> dict[str, torch.Tensor]:
        """Compute primal-dual loss from trajectory data."""
        # discounted rewards & costs
        returns = self._discount_cumsum(traj.rewards, gamma)
        cost_returns = self._discount_cumsum(traj.costs, gamma)

        # Normalize advantages
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)
        cost_adv = (cost_returns - cost_returns.mean()) / (cost_returns.std() + 1e-8)

        # Make sure the log_probs have a gradient path
        log_probs = traj.log_probs.squeeze()
        if not log_probs.requires_grad:
            # Recreate log probs from stored actions and observations through policy
            with torch.set_grad_enabled(True):
                batch_observations = traj.observations
                batch_actions = traj.actions
                action_distributions, _ = self.policy(batch_observations)
                log_probs = action_distributions.log_prob(batch_actions).sum(dim=1)

        # Policy loss is negative log probability times advantage
        policy_loss = -(log_probs * adv).mean()

        # Create constraint with grad_fn by using operations that maintain gradients
        cost_value = (log_probs * cost_adv).mean()  # This has gradients
        constraint = cost_value - self.cost_limit  # This will keep gradients

        # primal-dual Lagrangian
        loss = policy_loss + self.lambda_param.clamp(min=0) * constraint

        # Save metric history
        self.policy_loss_history.append(policy_loss.item())
        self.constraint_history.append(constraint.item())

        return dict(loss=loss, constraint=constraint, policy_loss=policy_loss, cost_value=cost_value)

    @staticmethod
    def _discount_cumsum(x: torch.Tensor, gamma: float):
        """Calculate discounted cumulative sum for returns calculation."""
        out = torch.zeros_like(x)
        running = 0.0
        for t in reversed(range(len(x))):
            running = x[t] + gamma * running
            out[t] = running
        return out

    def update(self, loss_dict: dict[str, torch.Tensor]):
        """Update policy parameters and Lagrange multiplier."""
        self.update_count += 1

        if self.meta_optimizer is None:
            self.opt.zero_grad()
            loss_dict["loss"].backward()
            self.opt.step()

            # lambda update (projected)
            with torch.no_grad():
                self.lambda_param.data = torch.clamp(
                    self.lambda_param.data + self.alpha_lambda * loss_dict["constraint"].detach(), min=0.0
                )
        elif self.is_pid_optimizer:
            # Compute reward and cost gradients
            policy_loss = loss_dict["policy_loss"]
            policy_loss.backward(retain_graph=True)
            reward_grads = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self.policy.parameters()
            ]
            for p in self.policy.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            constraint = loss_dict["constraint"]
            if constraint.requires_grad:
                constraint.backward()
                cost_grads = [
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in self.policy.parameters()
                ]
                for p in self.policy.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                cost_grads = [torch.zeros_like(p) for p in self.policy.parameters()]
            reward_grad_vec = torch.cat([g.flatten() for g in reward_grads])
            cost_grad_vec = torch.cat([g.flatten() for g in cost_grads])
            lambda_val = torch.tensor(self.lambda_param.item(), device=self.device)
            constraint_val = torch.tensor(constraint.item(), device=self.device)
            prev_constraint_val = (
                torch.tensor(self.constraint_history[-2], device=self.device)
                if len(self.constraint_history) > 1
                else None
            )
            # Call PIDLagrangianOptimizer forward (tracks history)
            delta_theta, delta_lambda = self.meta_optimizer(
                reward_grad_vec, cost_grad_vec, lambda_val, constraint_val, prev_constraint_val
            )
            # Apply policy update
            offset = 0
            for p in self.policy.parameters():
                n = p.numel()
                update = delta_theta[offset : offset + n].view_as(p)
                p.data.add_(update)
                offset += n
            # Update lambda
            new_lambda = self.lambda_param.data + delta_lambda
            self.lambda_param.data.copy_(torch.clamp(new_lambda, min=0.0))
            # Save lambda history for visualization
            self.lambda_history.append(self.lambda_param.item())
        else:
            # Compute separate gradients for reward and cost objectives
            # First, compute the reward gradient
            policy_loss = loss_dict["policy_loss"]
            policy_loss.backward(retain_graph=True)
            reward_grads = []
            for p in self.policy.parameters():
                if p.grad is not None:
                    reward_grads.append(p.grad.detach().clone())
                    p.grad.zero_()
                else:
                    reward_grads.append(torch.zeros_like(p))

            # Next, compute the cost gradient
            constraint = loss_dict["constraint"]
            if constraint.requires_grad:
                constraint.backward()  # No retain_graph needed
                cost_grads = []
                for p in self.policy.parameters():
                    if p.grad is not None:
                        cost_grads.append(p.grad.detach().clone())
                        p.grad.zero_()
                    else:
                        cost_grads.append(torch.zeros_like(p))
            else:
                cost_grads = [torch.zeros_like(p) for p in self.policy.parameters()]

            # Flatten gradients - make sure these don't require gradients for meta-optimizer input
            reward_grad_vec = torch.cat([g.flatten() for g in reward_grads]).detach()
            cost_grad_vec = torch.cat([g.flatten() for g in cost_grads]).detach()

            # Get expected input size
            expected_grad_dim = self.meta_optimizer.lstm.input_size

            # Store current values for PID controller
            constraint_value = constraint.item()  # Current constraint violation: J_C - beta

            # Check if this is the first update
            if not hasattr(self, "violation_history"):
                self.violation_history = [constraint_value]
                self.violation_integral = constraint_value
                self.prev_violation = 0.0
            else:
                self.violation_history.append(constraint_value)
                self.violation_integral += constraint_value
                self.prev_violation = self.violation_history[-2] if len(self.violation_history) > 1 else 0.0

            # Update meta-optimizer's integral and derivative terms
            self.meta_optimizer.violation_integral = torch.tensor(
                self.violation_integral, device=self.lambda_param.device
            )
            self.meta_optimizer.prev_violation = torch.tensor(self.prev_violation, device=self.lambda_param.device)

            # Create input vector for LSTM (can include history if needed)
            # Structure: [reward_grads, cost_grads, lambda, constraint_value]
            constraint_vec = torch.tensor([self.lambda_param.item(), constraint_value], device=reward_grad_vec.device)

            # Ensure the input has the correct dimension by truncating or padding
            total_grad_length = reward_grad_vec.numel() + cost_grad_vec.numel()
            if total_grad_length + constraint_vec.numel() != expected_grad_dim:
                # If dimensions don't match, adjust to expected dimension
                extra_dims = expected_grad_dim - (total_grad_length + constraint_vec.numel())
                if extra_dims > 0:
                    # Pad with zeros
                    padding = torch.zeros(extra_dims, device=reward_grad_vec.device)
                    input_vec = torch.cat([reward_grad_vec, cost_grad_vec, constraint_vec, padding])
                else:
                    # Truncate gradients to fit
                    # Shrink both reward and cost gradients proportionally
                    max_grad_length = (expected_grad_dim - constraint_vec.numel()) // 2
                    input_vec = torch.cat(
                        [reward_grad_vec[:max_grad_length], cost_grad_vec[:max_grad_length], constraint_vec]
                    )
            else:
                # Dimensions match, proceed normally
                input_vec = torch.cat([reward_grad_vec, cost_grad_vec, constraint_vec])

            # RNN expects (B, T, D) — here B=T=1
            # Form the input tensor as (B, T, D) where B=T=1
            input_tensor = input_vec.view(1, 1, -1)

            # Run forward pass through meta-optimizer - no gradients needed since we detached inputs
            with torch.no_grad():
                deltas, new_hx = self.meta_optimizer(input_tensor)

                # Log diagnostics for meta-optimizer visualization
                self.meta_optimizer.log_diagnostics(input_tensor, deltas, new_hx)

                # Get PID gains and learning rates from meta-optimizer
                pid_params = self.meta_optimizer.get_pid_parameters()
                k_p = pid_params["K_P"]
                k_i = pid_params["K_I"]
                k_d = pid_params["K_D"]
                eta_r = pid_params["eta_R"]
                eta_c = pid_params["eta_C"]

                # Apply policy update with structured form (F_phi)
                policy_params = list(self.policy.parameters())
                offset = 0

                # Use structured formula from the paper: η_R * ∇θJ_R - λ * η_C * ∇θJ_C
                for p in policy_params:
                    n = p.numel()
                    if offset + n <= reward_grad_vec.numel():  # Make sure we don't go out of bounds
                        # Extract reward gradient for this parameter
                        reward_grad = reward_grad_vec[offset : offset + n].view_as(p)
                        # Extract cost gradient for this parameter
                        cost_grad = cost_grad_vec[offset : offset + n].view_as(p)

                        # Apply the structured update formula (F_phi)
                        update = eta_r * reward_grad - self.lambda_param.item() * eta_c * cost_grad
                        p.data.add_(update)  # Apply update
                        offset += n
                    else:
                        break

                # Apply lambda update with PID formula (G_phi)
                violation = constraint_value
                lambda_update = (
                    k_p * violation + k_i * self.violation_integral + k_d * (violation - self.prev_violation)
                )
                new_lambda = self.lambda_param.data + lambda_update
                new_lambda = torch.clamp(new_lambda, min=0.0)  # Ensure non-negativity
                self.lambda_param.data.copy_(new_lambda)
        # Save lambda history for visualization
        self.lambda_history.append(self.lambda_param.item())

    def log_episode_metrics(self, traj: Trajectory) -> dict[str, float]:
        """Log and return a dictionary of episode metrics."""
        self.episode_count += 1
        metrics = traj.get_summary_metrics()

        # Add metrics specific to the primal-dual optimization
        metrics["lambda"] = self.lambda_param.item()
        metrics["constraint_violation"] = max(0, metrics["cost"] - self.cost_limit)
        metrics["episode_num"] = self.episode_count
        metrics["is_safe"] = metrics["cost"] <= self.cost_limit

        return metrics

    def visualize_lambda_history(self) -> plt.Figure:
        """Visualize the lambda parameter history over updates."""
        if not self.lambda_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No lambda history available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.lambda_history))
        ax.plot(x, self.lambda_history, "b-", linewidth=2)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("λ Value")
        ax.set_title("Lagrangian Multiplier (λ) Evolution")
        ax.grid(True, alpha=0.3)

        # If we have constraint history, add a second y-axis for violations
        if self.constraint_history:
            ax2 = ax.twinx()
            ax2.plot(
                np.arange(len(self.constraint_history)),
                self.constraint_history,
                "r--",
                alpha=0.7,
                label="Constraint Violation",
            )
            ax2.set_ylabel("Constraint Value", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

            # Add correlation coefficient between lambda and constraint
            if len(self.lambda_history) == len(self.constraint_history):
                corr = np.corrcoef(self.lambda_history, self.constraint_history)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.8),
                )

        fig.tight_layout()
        return fig

    def visualize_policy_distribution(self, obs_batch: torch.Tensor) -> plt.Figure:
        """Visualize the current policy's action distribution on a batch of observations."""
        with torch.no_grad():
            dist, _ = self.policy(obs_batch)
            means = dist.mean.cpu().numpy()
            stds = dist.scale.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot mean actions
        ax = axes[0]
        ax.boxplot(means)
        ax.set_title("Policy Mean Actions")
        ax.set_xlabel("Action Dimension")
        ax.set_ylabel("Mean Value")
        ax.grid(True, alpha=0.3)

        # Plot standard deviations
        ax = axes[1]
        ax.boxplot(stds)
        ax.set_title("Policy Action Std Deviations")
        ax.set_xlabel("Action Dimension")
        ax.set_ylabel("Std Value")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Policy Action Distribution Analysis", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def create_constraint_satisfaction_plot(self, cost_history: list[float]) -> plt.Figure:
        """Create a plot showing constraint satisfaction over time."""
        episodes = np.arange(1, len(cost_history) + 1)
        violations = [cost > self.cost_limit for cost in cost_history]
        violation_rate = sum(violations) / len(cost_history) if cost_history else 0

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot costs with color based on violation
        colors = ["green" if cost <= self.cost_limit else "red" for cost in cost_history]
        ax.scatter(episodes, cost_history, c=colors, alpha=0.7)

        # Connect the dots
        ax.plot(episodes, cost_history, "b-", alpha=0.3)

        # Add the constraint limit
        ax.axhline(y=self.cost_limit, color="k", linestyle="--", label=f"Cost Limit ({self.cost_limit})")

        # Add violation rate text
        ax.text(
            0.05,
            0.95,
            f"Violation Rate: {violation_rate * 100:.1f}%",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.set_title("Constraint Satisfaction Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
