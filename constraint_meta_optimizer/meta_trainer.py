import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
import seaborn as sns
import torch
from torch.amp import GradScaler

from constraint_meta_optimizer.agent import PrimalDualAgent, Trajectory
from constraint_meta_optimizer.envs import make_vec_env
from constraint_meta_optimizer.logger import ResearchLogger
from constraint_meta_optimizer.meta_optimizer import MetaOptimizerRNN, PIDLagrangianOptimizer

# Set of colors for plotting different tasks
TASK_COLORS = {
    "SafetyPointGoal1-v0": "#1f77b4",  # blue
    "SafetyPointGoal2-v0": "#ff7f0e",  # orange
    "SafetyCarGoal1-v0": "#2ca02c",  # green
    "SafetyCarGoal2-v0": "#d62728",  # red
    "SafetyAntGoal1-v0": "#9467bd",  # purple
    "SafetyAntGoal2-v0": "#8c564b",  # brown
}


class MetaTrainer:
    """Bilevel meta-training loop."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Device setup
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # Enable mixed precision for faster training
        self.use_mixed_precision = bool(torch.cuda.is_available())
        self.scaler = GradScaler("cuda") if self.use_mixed_precision else None
        print(f"🚀 Mixed precision training: {self.use_mixed_precision}")

        # Seed everything
        self._set_seeds(self.cfg.seed)
        print(f"🔢 Random seed: {self.cfg.seed}")

        # Initialize a temporary agent to get the exact parameter count
        safe_env = safety_gymnasium.make("SafetyPointGoal1-v0")
        env = SafetyGymnasium2Gymnasium(safe_env)
        temp_agent = PrimalDualAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            cost_limit=self.cfg.cost_limit,
            meta_optimizer=None,
            device=self.device,
        )

        # Calculate the exact gradient dimension based on the actual policy
        self.grad_dim = sum(p.numel() for p in temp_agent.policy.parameters()) + 2  # +2 for lambda and constraint
        print(f"📊 Policy parameter count: {self.grad_dim - 2}")

        # Create the meta-optimizer with the correct dimension
        if getattr(self.cfg, "meta_optimizer_type", "lstm").lower() == "pid":
            self.meta_opt = PIDLagrangianOptimizer(
                cost_limit=self.cfg.cost_limit,
                init_KP=self.cfg.init_KP,
                init_KI=self.cfg.init_KI,
                init_KD=self.cfg.init_KD,
                init_etaR=self.cfg.init_etaR,
                init_etaC=self.cfg.init_etaC,
                device=self.device,
            ).to(self.device)
            print("🧠 Meta-optimizer: PID Lagrangian (structured form)")
        else:
            self.meta_opt = MetaOptimizerRNN(
                grad_dim=self.grad_dim, hidden_size=self.cfg.hidden_size, num_layers=self.cfg.lstm_layers
            ).to(self.device)
            try:
                if hasattr(torch, "compile"):
                    self.meta_opt = torch.compile(self.meta_opt)
                    print("🚀 Using torch.compile for the meta-optimizer")
            except Exception as e:
                print(f"⚠️ Could not use torch.compile: {e}")
            print(f"🧠 Meta-optimizer: LSTM with {self.cfg.hidden_size} hidden units, {self.cfg.lstm_layers} layers")

        self.outer_optim = torch.optim.Adam(self.meta_opt.parameters(), lr=self.cfg.meta_lr)
        print(f"🧠 Meta-optimizer: LSTM with {self.cfg.hidden_size} hidden units, {self.cfg.lstm_layers} layers")

        # Logging setup
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = Path("runs") / self.cfg.meta_optimizer_type / timestamp

        # Create metadata for the experiment
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)

        metadata = {
            "experiment": "Constraint-Aware Meta-Optimizer",
            "timestamp": timestamp,
            "config": config_dict,
            "device": str(self.device),
            "policy_params": self.grad_dim - 2,
            "environments": [str(env_id) for env_id in self.cfg.env_ids],  # Ensure all values are JSON serializable
            "meta_optimizer": {
                "type": "LSTM",
                "hidden_size": int(self.cfg.hidden_size),  # Ensure numeric values are properly converted
                "num_layers": int(self.cfg.lstm_layers),
            },
        }

        self.logger = ResearchLogger(
            logdir=logdir,
            experiment_name=None,  # Use default timestamp
            metadata=metadata,
        )

        # Metric tracking
        self.task_returns = {
            str(env_id): [] for env_id in self.cfg.env_ids
        }  # Convert keys to strings for JSON compatibility
        self.task_costs = {str(env_id): [] for env_id in self.cfg.env_ids}
        self.meta_losses = []
        self.constraint_violations = []
        self.episode_lengths = []
        self.start_time = time.time()

        print("📝 Configuration:")
        print(OmegaConf.to_yaml(self.cfg))
        # Starting meta-training info
        env_list = ", ".join(str(env) for env in self.cfg.env_ids)
        print(f"🔍 Starting meta-training with {len(self.cfg.env_ids)} environments:")
        print(f"    {env_list}")

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self) -> None:
        """Execute meta-training loop with advanced tracking and live progress."""
        # best_mean_return for tracking
        best_mean_return = float("-inf")
        total_start_time = time.time()
        env_step_time = 0.0
        model_step_time = 0.0
        print("🏁 Starting meta-training...")
        for meta_iter in range(self.cfg.total_meta_iters):
            meta_iter_start = time.time()

            # Sample tasks for this meta-iteration
            tasks = random.choices(self.cfg.env_ids, k=self.cfg.meta_batch_size)

            # Track per-task metrics for this iteration
            iter_metrics = {
                "meta_loss": 0.0,
                "returns": [],
                "costs": [],
                "violations": [],
                "task_metrics": {task: {"returns": [], "costs": [], "violations": []} for task in tasks},
            }

            # Collect all meta-objectives for the meta-loss computation
            all_meta_losses = []

            # Run inner loop for each task
            # Time breakdown
            env_time = 0.0
            model_time = 0.0
            for task in tasks:
                inner_losses, final_metrics = self._run_inner_loop(task)
                env_time += final_metrics.get("env_time", 0.0)
                model_time += final_metrics.get("model_time", 0.0)
                mean_return = final_metrics.get("mean_return", 0)
                mean_cost = final_metrics.get("mean_cost", 0)
                constraint_violation = final_metrics.get("constraint_violation", 0)

                # Compute meta-objective with penalty: U(φ) = J_R(π_θ_T) - μ * max(0, J_C(π_θ_T) - β)
                meta_obj = mean_return - self.cfg.mu * max(0, mean_cost - self.cfg.cost_limit)

                # Store task-specific metrics
                self.task_returns[task].append(mean_return)
                self.task_costs[task].append(mean_cost)
                self.constraint_violations.append(constraint_violation)

                # Update iteration metrics
                iter_metrics["returns"].append(mean_return)
                iter_metrics["costs"].append(mean_cost)
                iter_metrics["violations"].append(constraint_violation)
                iter_metrics["task_metrics"][task]["returns"].append(mean_return)
                iter_metrics["task_metrics"][task]["costs"].append(mean_cost)
                iter_metrics["task_metrics"][task]["violations"].append(constraint_violation)

                # Add meta-objective for this task (negative because we're minimizing)
                task_meta_loss = torch.tensor(-meta_obj, device=self.device, dtype=torch.float32)
                all_meta_losses.append(task_meta_loss)

                # If we have inner losses to incorporate, add them too
                if inner_losses:
                    # Stack and average inner losses if there are multiple
                    inner_loss_tensor = torch.stack(inner_losses).mean()
                    all_meta_losses.append(inner_loss_tensor)

            env_step_time += env_time
            model_step_time += model_time

            # Reset the meta-optimizer hidden state before computing meta-loss
            # This ensures we don't carry over computational graph between tasks
            self.meta_opt.reset_hidden_state()

            # Compute meta-loss as average across all collected losses
            if all_meta_losses:
                # Detach and recreate tensors to avoid double backward issues
                detached_losses = [loss.detach().clone().requires_grad_(True) for loss in all_meta_losses]
                meta_loss = torch.stack(detached_losses).mean()
            else:
                # If no losses were collected, create a default loss tensor
                meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            iter_metrics["meta_loss"] = meta_loss.item()

            # Meta-optimizer update with GPU stats logging
            print(f"🧠 Meta-Update for iteration {meta_iter} | Loss: {meta_loss.item():.4f}")

            # Check and report GPU memory usage
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                    print(f"  💾 GPU Memory: {gpu_memory:.2f} GB")

                    # Try to get GPU utilization if pynvml is available
                    try:
                        gpu_utilization = torch.cuda.utilization()
                        print(f"  💾 GPU Utilization: {gpu_utilization}%")
                    except (ModuleNotFoundError, AttributeError):
                        # Skip utilization reporting if pynvml is not available
                        pass
                except Exception as e:
                    print(f"  ⚠️ Error reporting GPU stats: {e}")

            # Perform FP32 meta-update (disable AMP for outer optimizer)
            self.outer_optim.zero_grad()
            meta_loss.backward()  # Compute gradients
            torch.nn.utils.clip_grad_norm_(self.meta_opt.parameters(), 1.0)
            self.outer_optim.step()

            # Print current PID parameters
            pid_params = (
                self.meta_opt.get_meta_parameters()
                if hasattr(self.meta_opt, "get_meta_parameters")
                else self.meta_opt.get_pid_parameters()
            )
            print(
                f"  🎛️ K_P: {pid_params.get('K_P', 0):.4f} | "
                f"K_I: {pid_params.get('K_I', 0):.4f} | "
                f"K_D: {pid_params.get('K_D', 0):.4f} | "
                f"η_R: {pid_params.get('eta_R', 0):.6f} | "
                f"η_C: {pid_params.get('eta_C', 0):.6f}"
            )

            # Store meta-loss
            self.meta_losses.append(meta_loss.item())

            # Compute mean metrics across tasks
            mean_return = np.mean(iter_metrics["returns"])
            mean_cost = np.mean(iter_metrics["costs"])
            mean_violation = np.mean(iter_metrics["violations"])

            # Log metrics
            iter_time = time.time() - meta_iter_start
            self.logger.log(
                step=meta_iter,
                meta_loss=meta_loss.item(),
                mean_return=mean_return,
                mean_cost=mean_cost,
                constraint_violation_rate=mean_violation,
                iteration_time=iter_time,
            )

            # After each meta-iteration, print detailed progress
            print(
                f"\n[Meta-Iter {meta_iter}] Meta-Loss: {meta_loss.item():.4f} | "
                f"Mean Return: {mean_return:.2f} | Mean Cost: {mean_cost:.2f} | "
                f"Violation Rate: {mean_violation * 100:.1f}% | "
                f"Time: {iter_time:.2f}s | Env Step: {env_time:.2f}s | Model Step: {model_time:.2f}s"
            )
            for task in tasks:
                print(
                    f"    Task {task}: Return={iter_metrics['task_metrics'][task]['returns'][-1]:.2f}, "
                    f"Cost={iter_metrics['task_metrics'][task]['costs'][-1]:.2f}, "
                    f"Violation={iter_metrics['task_metrics'][task]['violations'][-1]:.2f}"
                )

            # Live progress print
            elapsed = time.time() - total_start_time
            iters_left = self.cfg.total_meta_iters - (meta_iter + 1)
            eta = (elapsed / (meta_iter + 1)) * iters_left if meta_iter > 0 else 0
            bar_len = 30
            filled_len = int(bar_len * (meta_iter + 1) / self.cfg.total_meta_iters)
            bar = "=" * filled_len + "-" * (bar_len - filled_len)
            sys.stdout.write(
                f"\r[{bar}] {meta_iter + 1}/{self.cfg.total_meta_iters} | "
                f"Return: {mean_return:.2f} | Cost: {mean_cost:.2f} | Viol: {mean_violation * 100:.1f}% | "
                f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m | "
                f"Env: {env_time:.1f}s | Model: {model_time:.1f}s\n\n"
            )
            sys.stdout.flush()

            # Create and log visualizations periodically
            if meta_iter % self.cfg.log_interval == 0:
                print()  # Newline after progress bar
                # Generate training curves
                self._generate_training_visualizations(meta_iter)

                # Save model checkpoint
                self._save_checkpoint(meta_iter)

                # Check for new best model
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    self._save_checkpoint(meta_iter, is_best=True)

                # Compute ETA and progress
                elapsed_time = time.time() - self.start_time
                steps_per_sec = (meta_iter + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_iters = self.cfg.total_meta_iters - meta_iter - 1
                eta_seconds = remaining_iters / steps_per_sec if steps_per_sec > 0 else 0

                # Print detailed progress
                print(
                    f"📊 Meta-Iter {meta_iter}/{self.cfg.total_meta_iters} "
                    f"[{meta_iter / self.cfg.total_meta_iters * 100:.1f}%] | "
                    f"Meta-Loss: {meta_loss.item():.4f} | "
                    f"Mean Return: {mean_return:.2f} | "
                    f"Mean Cost: {mean_cost:.2f} | "
                    f"Violation Rate: {mean_violation * 100:.1f}% | "
                    f"ETA: {self._format_time(eta_seconds)}\n"
                )

                # Flush logger to ensure data is written
                self.logger.flush()

        print("\n✅ Meta-training complete!")
        total_time = time.time() - total_start_time
        print(
            f"[SUMMARY] Total time: {total_time / 60:.1f} min | "
            f"Env step: {env_step_time / 60:.1f} min | "
            f"Model step: {model_step_time / 60:.1f} min"
        )
        if model_step_time / (total_time + 1e-8) < 0.1:
            print(
                "[WARNING] Less than 10% of time spent in model step. "
                "This is typical for RL, but if you want higher GPU utilization, "
                "use vectorized environments (see README_OPTIMIZATIONS.md)."
            )
        self._generate_training_visualizations(self.cfg.total_meta_iters, final=True)
        self._save_checkpoint(self.cfg.total_meta_iters, is_final=True)
        self.logger.close()

    def _run_inner_loop(self, env_id: str) -> tuple[list, dict]:
        """Run inner loop for a single task with detailed metrics.

        Implements the bi-level training procedure from the paper:
        1. Inner loop (policy training): Initialize θ_0 and λ_0, then apply updates for T steps
        2. Meta-objective: U(φ) = J_R(π_θ_T) - μ * max(0, J_C(π_θ_T) - β)
        """
        self.meta_opt.reset_hidden_state()
        num_envs = max(getattr(self.cfg, "vector_envs_per_task", 1), 2)  # Always use at least 2 envs
        max_steps = self.cfg.episode_length
        inner_rollouts = self.cfg.inner_rollouts
        traj_losses = []
        env = make_vec_env(env_id, num_envs)
        agent = PrimalDualAgent(
            obs_dim=env.single_observation_space.shape[0],
            act_dim=env.single_action_space.shape[0],
            cost_limit=self.cfg.cost_limit,
            meta_optimizer=self.meta_opt,
            device=self.device,
        )
        obs, _ = env.reset(seed=None)
        ep_rews = np.zeros(num_envs)
        ep_costs = np.zeros(num_envs)
        ep_lens = np.zeros(num_envs, dtype=int)
        episode_returns = []
        episode_costs = []
        for _ in range(inner_rollouts):
            # Collect one rollout per environment
            all_observations = [[] for _ in range(num_envs)]
            all_actions = [[] for _ in range(num_envs)]
            all_rewards = [[] for _ in range(num_envs)]
            all_costs = [[] for _ in range(num_envs)]
            all_log_probs = [[] for _ in range(num_envs)]
            ep_complete = np.zeros(num_envs, dtype=bool)
            step_count = 0
            while not np.all(ep_complete) and step_count < max_steps:
                actions_and_logprobs = [agent.select_action(o, return_log_prob=True) for o in obs]
                acts = np.stack([item[0] for item in actions_and_logprobs])
                log_ps = np.array([item[1] for item in actions_and_logprobs])
                next_obs, rews, terms, truncs, infos = env.step(acts)
                # Extract costs
                if isinstance(infos, dict) and "cost" in infos:
                    costs = np.array(infos["cost"])
                elif isinstance(infos, (list, tuple)) and all(isinstance(info, dict) for info in infos):
                    costs = np.array([info.get("cost", 0.0) for info in infos])
                else:
                    costs = np.zeros(num_envs)
                # Record data
                for i in range(num_envs):
                    if not ep_complete[i]:
                        all_observations[i].append(obs[i])
                        all_actions[i].append(acts[i])
                        all_rewards[i].append(rews[i])
                        all_costs[i].append(costs[i])
                        all_log_probs[i].append(log_ps[i])
                        ep_rews[i] += rews[i]
                        ep_costs[i] += costs[i]
                        ep_lens[i] += 1
                done_this_step = np.logical_or(terms, truncs)
                ep_complete = np.logical_or(ep_complete, done_this_step)
                obs = next_obs
                step_count += 1
                if np.any(done_this_step) and not np.all(ep_complete):
                    reset_indices = np.where(done_this_step)[0]
                    for idx in reset_indices:
                        if done_this_step[idx]:
                            episode_returns.append(ep_rews[idx])
                            episode_costs.append(ep_costs[idx])
                            ep_rews[idx] = 0
                            ep_costs[idx] = 0
                            ep_lens[idx] = 0
            # After rollout, update policy and meta-optimizer
            for i in range(num_envs):
                if all_observations[i]:
                    # Build trajectory and compute/update
                    traj = Trajectory(
                        observations=torch.tensor(
                            np.array(all_observations[i]), dtype=torch.float32, device=self.device
                        ),
                        actions=torch.tensor(np.array(all_actions[i]), dtype=torch.float32, device=self.device),
                        rewards=torch.tensor(np.array(all_rewards[i]), dtype=torch.float32, device=self.device),
                        costs=torch.tensor(np.array(all_costs[i]), dtype=torch.float32, device=self.device),
                        log_probs=torch.tensor(all_log_probs[i], dtype=torch.float32, device=self.device),
                    )
                    loss_dict = agent.compute_loss(traj)
                    agent.update(loss_dict)
                    traj_losses.append(loss_dict["loss"])
        # Collect any remaining episode returns for metrics
        for i in range(num_envs):
            if ep_lens[i] > 0:
                episode_returns.append(ep_rews[i])
                episode_costs.append(ep_costs[i])
        mean_return = np.mean(episode_returns[-self.cfg.unroll_steps :])
        mean_cost = np.mean(episode_costs[-self.cfg.unroll_steps :])
        constraint_violation = np.mean([cost > self.cfg.cost_limit for cost in episode_costs[-self.cfg.unroll_steps :]])
        mu = 10.0
        cost_violation = max(0, mean_cost - self.cfg.cost_limit)
        meta_objective = mean_return - mu * cost_violation
        task_metrics = {
            "mean_return": mean_return,
            "mean_cost": mean_cost,
            "constraint_violation": constraint_violation,
            "returns": episode_returns,
            "costs": episode_costs,
            "meta_objective": meta_objective,
            "cost_violation": cost_violation,
        }
        env.close()
        return traj_losses, task_metrics

    def _generate_training_visualizations(self, step: int, final: bool = False) -> None:
        """Generate and log publication-quality visualizations."""
        # Set the style for figures
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # 1. Meta-Loss Plot
        if self.meta_losses:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.meta_losses, "b-", alpha=0.7)
            ax.plot(np.convolve(self.meta_losses, np.ones(10) / 10, mode="valid"), "r-", linewidth=2)
            ax.set_xlabel("Meta-Iteration")
            ax.set_ylabel("Meta-Loss")
            ax.set_title("Meta-Optimizer Loss During Training")
            ax.grid(True, alpha=0.3)
            self.logger.log_image("meta_loss", fig, step)

        # 2. Task Returns Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for env_id, returns in self.task_returns.items():
            if returns:
                color = TASK_COLORS.get(env_id, "gray")
                ax.plot(returns, label=env_id, alpha=0.7, color=color)
        ax.set_xlabel("Meta-Iteration")
        ax.set_ylabel("Episode Return")
        ax.set_title("Returns Across Tasks")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self.logger.log_image("task_returns", fig, step)

        # 3. Task Costs and Constraint Violations
        fig, ax = plt.subplots(figsize=(10, 6))
        for env_id, costs in self.task_costs.items():
            if costs:
                color = TASK_COLORS.get(env_id, "gray")
                ax.plot(costs, label=env_id, alpha=0.7, color=color)
        ax.axhline(y=self.cfg.cost_limit, color="r", linestyle="--", label=f"Cost Limit ({self.cfg.cost_limit})")
        ax.set_xlabel("Meta-Iteration")
        ax.set_ylabel("Episode Cost")
        ax.set_title("Safety Costs Across Tasks")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self.logger.log_image("task_costs", fig, step)

        # 4. Constraint Violation Rate
        if self.constraint_violations:
            fig, ax = plt.subplots(figsize=(10, 6))
            window_size = min(10, len(self.constraint_violations))
            if window_size > 1:
                smoothed = np.convolve(self.constraint_violations, np.ones(window_size) / window_size, mode="valid")
                ax.plot(self.constraint_violations, "o", alpha=0.3, color="gray")
                ax.plot(range(window_size - 1, len(self.constraint_violations)), smoothed, "b-", linewidth=2)
            else:
                ax.plot(self.constraint_violations, "b-")
            ax.set_xlabel("Meta-Iteration")
            ax.set_ylabel("Constraint Violation Rate")
            ax.set_title("Safety Constraint Violations")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            self.logger.log_image("constraint_violations", fig, step)

        # 5. Combined Return, Cost, Violation Plot
        if self.task_returns and self.task_costs and self.constraint_violations:
            # Find the minimum length of data across all tasks to avoid index errors
            min_length = min([len(returns) for returns in self.task_returns.values()])
            min_length = min(min_length, len(self.constraint_violations))

            if min_length > 0:
                # Safely compute mean metrics across tasks per iteration up to min_length
                mean_returns = np.array(
                    [
                        np.mean(
                            [self.task_returns[env][i] for env in self.task_returns if i < len(self.task_returns[env])]
                        )
                        for i in range(min_length)
                    ]
                )
                mean_costs = np.array(
                    [
                        np.mean([self.task_costs[env][i] for env in self.task_costs if i < len(self.task_costs[env])])
                        for i in range(min_length)
                    ]
                )
                mean_violations = np.array(self.constraint_violations[:min_length])
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(mean_returns, "b-", label="Mean Return")
                ax1.plot(mean_costs, "r-", label="Mean Cost")
                ax1.set_xlabel("Meta-Iteration")
                ax1.set_ylabel("Return / Cost")
                ax2 = ax1.twinx()
                ax2.plot(mean_violations, "g--", label="Violation Rate")
                ax2.set_ylabel("Violation Rate")
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, loc="best")
                ax1.set_title("Mean Return, Cost, and Constraint Violation Over Meta-Training")
                ax1.grid(True, alpha=0.3)
                self.logger.log_image("combined_metrics", fig, step)

        # 6. Metric Correlation Scatter Plots
        if (
            self.task_returns
            and self.task_costs
            and self.constraint_violations
            and "min_length" in locals()
            and min_length > 0
        ):  # Create a more focused figure showing the Pareto-optimal frontier
            fig, ax = plt.subplots(figsize=(10, 8))

            # Calculate meta-training phase (early, mid, late)
            n_points = len(mean_returns)
            phases = np.zeros(n_points, dtype=int)
            phases[: n_points // 3] = 0  # Early phase
            phases[n_points // 3 : 2 * n_points // 3] = 1  # Mid phase
            phases[2 * n_points // 3 :] = 2  # Late phase

            # Marker size based on violation rate (smaller = better)
            sizes = 100 * (1 - mean_violations)
            sizes = np.maximum(sizes, 10)  # Ensure minimum visibility

            # Use distinct markers for different training phases
            markers = ["o", "s", "^"]
            phases_labels = ["Early Training", "Mid Training", "Late Training"]

            # Plot Return vs Cost with training phase and violation information
            for i, phase in enumerate([0, 1, 2]):
                mask = phases == phase
                if not any(mask):
                    continue

                scatter = ax.scatter(
                    mean_costs[mask],
                    mean_returns[mask],
                    s=sizes[mask],
                    c=mean_violations[mask],
                    alpha=0.7,
                    marker=markers[i],
                    cmap="RdYlGn_r",  # Red (high violation) to Green (low violation)
                    label=phases_labels[i],
                )

            # Add cost limit line
            ax.axvline(x=self.cfg.cost_limit, color="r", linestyle="--", label=f"Cost Limit ({self.cfg.cost_limit})")

            # Highlight Pareto frontier if we have enough points
            if n_points > 10:
                # Find approximate Pareto front points (lower cost, higher return is better)
                # Using simple approach - can be refined with proper Pareto calculation
                pareto_indices = []
                sorted_by_cost = np.argsort(mean_costs)
                max_return = -float("inf")
                for idx in sorted_by_cost:
                    if mean_returns[idx] > max_return:
                        pareto_indices.append(idx)
                        max_return = mean_returns[idx]

                # Sort pareto points by cost for line drawing
                pareto_indices.sort(key=lambda i: mean_costs[i])

                if pareto_indices:
                    pareto_costs = mean_costs[pareto_indices]
                    pareto_returns = mean_returns[pareto_indices]
                    ax.plot(pareto_costs, pareto_returns, "k--", linewidth=2, label="Approx. Pareto Front")

            # Add colorbar for violation rate
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Constraint Violation Rate")

            ax.set_xlabel("Mean Cost")
            ax.set_ylabel("Mean Return")
            ax.set_title("Return-Cost Trade-off with Safety Constraint Violations")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            self.logger.log_image("pareto_tradeoff", fig, step)

            # Creating a radar chart for multi-dimensional performance view
            if n_points > 0:
                # Select key points from training: beginning, 1/3, 2/3, and final
                indices = [0, n_points // 3, 2 * n_points // 3, -1]
                indices = [i if i >= 0 else n_points + i for i in indices]
                indices = [min(i, n_points - 1) for i in indices]  # Ensure within bounds

                # For radar chart, prepare metrics: higher is better
                # For cost and violation: invert so lower values are better on the chart
                normalized_returns = mean_returns[indices] / max(mean_returns.max(), 1e-8)
                normalized_costs = 1 - (mean_costs[indices] / max(mean_costs.max(), 1e-8))  # Invert
                normalized_violations = 1 - mean_violations[indices]  # Invert

                # Calculate meta-objective and safety margin
                meta_objectives = mean_returns[indices] - 10.0 * np.maximum(
                    0, mean_costs[indices] - self.cfg.cost_limit
                )
                normalized_meta_obj = meta_objectives / max(abs(meta_objectives).max(), 1e-8)
                normalized_meta_obj = (normalized_meta_obj + 1) / 2  # Scale to [0,1]

                safety_margins = np.maximum(0, self.cfg.cost_limit - mean_costs[indices]) / self.cfg.cost_limit

                # Prepare radar chart data
                labels = ["Return", "Cost Efficiency", "Safety Compliance", "Meta-Objective", "Safety Margin"]
                metrics = np.column_stack(
                    (
                        normalized_returns,
                        normalized_costs,
                        normalized_violations,
                        normalized_meta_obj,
                        safety_margins,
                    )
                )

                # Create radar chart
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

                # Plot each training phase
                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop

                # Ensure proper labeling
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.degrees(angles[:-1]), labels)

                for i, idx in enumerate(indices):
                    values = metrics[i].tolist()
                    values += values[:1]  # Close the loop

                    phase_idx = 0 if idx < n_points // 3 else (1 if idx < 2 * n_points // 3 else 2)
                    label = f"Iteration {idx} ({phases_labels[phase_idx]})"

                    ax.plot(angles, values, linewidth=2, label=label)
                    ax.fill(angles, values, alpha=0.1)

                ax.set_title(
                    f"Multi-dimensional Performance Evolution\n({self.cfg.meta_optimizer_type.upper()} Meta-Optimizer)"
                )
                ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
                self.logger.log_image("performance_radar", fig, step)

                # Training trajectory visualization in 3D
                if n_points >= 10:  # Only if we have enough data points
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection="3d")

                    # Show progression with color gradient
                    norm = plt.Normalize(0, n_points - 1)
                    plt.cm.viridis(norm(range(n_points)))

                    # Create scatter plot with trajectory line
                    scatter = ax.scatter(
                        mean_returns, mean_costs, mean_violations, c=range(n_points), cmap="viridis", s=50, alpha=0.8
                    )

                    # Add connecting line to show trajectory
                    ax.plot(mean_returns, mean_costs, mean_violations, "k-", alpha=0.3)

                    # Mark start and end points
                    ax.scatter(mean_returns[0], mean_costs[0], mean_violations[0], color="blue", s=100, label="Start")
                    ax.scatter(mean_returns[-1], mean_costs[-1], mean_violations[-1], color="red", s=100, label="End")

                    # Add cost limit plane
                    xlim = ax.get_xlim()
                    (0, max(self.cfg.cost_limit * 2, mean_costs.max() * 1.2))
                    zlim = ax.get_zlim()

                    # Create cost limit plane
                    x_plane = np.linspace(xlim[0], xlim[1], 10)
                    z_plane = np.linspace(zlim[0], zlim[1], 10)
                    x_plane, z_plane = np.meshgrid(x_plane, z_plane)
                    y_plane = np.ones_like(x_plane) * self.cfg.cost_limit

                    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color="r")

                    # Customize labels and title
                    ax.set_xlabel("Return")
                    ax.set_ylabel("Cost")
                    ax.set_zlabel("Violation Rate")
                    ax.set_title(
                        f"3D Training Trajectory in Return-Cost-Violation Space\n({self.cfg.meta_optimizer_type.upper()} Meta-Optimizer)"
                    )

                    # Add colorbar to show progression
                    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                    cbar.set_label("Training Progress (Meta-Iterations)")

                    ax.legend()
                    self.logger.log_image("training_trajectory_3d", fig, step)

        # Constraint Violation Histogram for final analysis
        if final and self.constraint_violations:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.constraint_violations, bins=20, kde=True, ax=ax)
            ax.axvline(
                x=np.mean(self.constraint_violations),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(self.constraint_violations):.3f}",
            )
            ax.set_xlabel("Constraint Violation Rate")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Constraint Violations")
            ax.legend()
            self.logger.log_image("constraint_violation_dist", fig, step)

        # 7. Meta-Optimizer Dynamics Plots (PID, updates, gradients, etc.)
        if hasattr(self.meta_opt, "visualize_pid_gains"):
            fig = self.meta_opt.visualize_pid_gains()
            self.logger.log_image("pid_gain_evolution", fig, step)
        if hasattr(self.meta_opt, "visualize_update_magnitudes"):
            fig = self.meta_opt.visualize_update_magnitudes()
            self.logger.log_image("update_magnitudes", fig, step)
        if hasattr(self.meta_opt, "visualize_grad_norms"):
            fig = self.meta_opt.visualize_grad_norms()
            self.logger.log_image("gradient_norms", fig, step)
        if hasattr(self.meta_opt, "visualize_theta_lambda_updates"):
            fig = self.meta_opt.visualize_theta_lambda_updates()
            self.logger.log_image("theta_lambda_updates", fig, step)
        if hasattr(self.meta_opt, "visualize_hidden_evolution"):
            fig = self.meta_opt.visualize_hidden_evolution()
            self.logger.log_image("hidden_state_evolution", fig, step)

        # 8. Save a sample trajectory plot for qualitative analysis (if available)
        if final and hasattr(self, "task_returns") and self.task_returns:
            # Try to get the last environment used
            last_env = list(self.task_returns.keys())[-1]
            try:
                # Re-run one episode to get a trajectory plot
                safe_env = safety_gymnasium.make(last_env)
                env = SafetyGymnasium2Gymnasium(safe_env)
                agent = PrimalDualAgent(
                    obs_dim=env.observation_shape[0],
                    act_dim=env.action_space.shape[0],
                    cost_limit=self.cfg.cost_limit,
                    meta_optimizer=self.meta_opt,
                    device=self.device,
                )
                obs, _ = env.reset(seed=None)
                obs_list, act_list, rew_list, cost_list = [], [], [], []
                done = False
                steps = 0
                while not done and steps < self.cfg.episode_length:
                    act = agent.select_action(obs)
                    next_obs, rew, term, trunc, info = env.step(act)
                    cost = info.get("cost", 0.0)
                    obs_list.append(obs)
                    act_list.append(act)
                    rew_list.append(rew)
                    cost_list.append(cost)
                    obs = next_obs
                    done = term or trunc
                    steps += 1
                traj = Trajectory(
                    observations=torch.tensor(np.array(obs_list), dtype=torch.float32),
                    actions=torch.tensor(np.array(act_list), dtype=torch.float32),
                    rewards=torch.tensor(np.array(rew_list), dtype=torch.float32),
                    costs=torch.tensor(np.array(cost_list), dtype=torch.float32),
                    log_probs=torch.zeros(len(obs_list)),
                )
                fig = traj.visualize(step=step)
                self.logger.log_image("sample_trajectory", fig, step)
            except Exception as e:
                print(f"[WARN] Could not generate sample trajectory plot: {e}")

        # 9. Save final performance metrics (mean/std reward/cost) to CSV/JSON for report inclusion
        if final:
            perf_data = {}
            for env_id in self.task_returns:
                returns = np.array(self.task_returns[env_id])
                costs = np.array(self.task_costs[env_id])
                perf_data[env_id] = {
                    "mean_return": float(np.mean(returns)) if len(returns) else None,
                    "std_return": float(np.std(returns)) if len(returns) else None,
                    "mean_cost": float(np.mean(costs)) if len(costs) else None,
                    "std_cost": float(np.std(costs)) if len(costs) else None,
                }
            # Save as JSON
            perf_path = self.logger.logdir / "final_performance.json"
            with open(perf_path, "w", encoding="utf-8") as f:
                json.dump(perf_data, f, indent=2)
            # Save as CSV
            csv_path = self.logger.logdir / "final_performance.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["env_id", "mean_return", "std_return", "mean_cost", "std_cost"])
                for env_id, vals in perf_data.items():
                    writer.writerow(
                        [
                            env_id,
                            vals["mean_return"],
                            vals["std_return"],
                            vals["mean_cost"],
                            vals["std_cost"],
                        ]
                    )

        # Close all figures to prevent memory leaks
        plt.close("all")

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.logger.logdir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "meta_optimizer_state_dict": self.meta_opt.state_dict(),
            "optimizer_state_dict": self.outer_optim.state_dict(),
            "grad_dim": self.grad_dim,
            "hidden_size": self.cfg.hidden_size,
            "num_layers": self.cfg.lstm_layers,
            "step": step,
            "config": OmegaConf.to_container(self.cfg),
        }

        already_saved = False

        # Save periodic checkpoint
        if step % self.cfg.checkpoint_interval == 0 or is_final:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint to {checkpoint_path}")
            already_saved = True

        # Save best model so far
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            # Only print the checkpoint_{step} message if we didn't already save it above
            if not already_saved:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"💾 Saved checkpoint to {checkpoint_path}")

            print(f"🏆 New best model saved to {best_path}")

        # Always save latest model
        latest_path = checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"


@hydra.main(version_base="1.3", config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run meta-training for constraint-aware meta-optimizer."""
    print("=" * 80)
    print("🔬 Constraint-Aware Meta-Optimizer for Policy Gradient")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("\n" + "=" * 80)

    trainer = MetaTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
