import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import safety_gymnasium  # noqa: F401


def make_env(env_id: str) -> gym.Env:
    """Create a single Gymnasium environment."""
    return gym.make(env_id)


def make_vec_env(env_id: str, num_envs: int) -> SyncVectorEnv:
    """Create a vectorized environment for parallel rollout."""

    def make_single_env() -> gym.Env:
        safe_env = safety_gymnasium.make(env_id)
        return safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)

    return SyncVectorEnv([make_single_env for _ in range(num_envs)])
