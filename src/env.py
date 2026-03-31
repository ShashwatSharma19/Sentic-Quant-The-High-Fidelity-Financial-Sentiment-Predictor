"""
env.py
------
A custom Gymnasium environment for trading.
Allows Reinforcement Learning agents to safely interact with market sequences.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Standard Trading Environment.
    State: Active market technical indicators + current position.
    Actions: 0 (Short), 1 (Flat), 2 (Long).
    Reward: Daily percentage step return minus transaction costs.
    """
    def __init__(self, features: np.ndarray, prices: np.ndarray):
        super().__init__()
        self.features = features
        self.prices = prices
        self.n_steps = len(features)
        
        # Observation Box: [feature_1, feature_2, ..., feature_n, current_position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(features.shape[1] + 1,), dtype=np.float32
        )
        
        # Action Discrete: 3 potential moves
        self.action_space = spaces.Discrete(3)
        
        self.current_step = 0
        self.position = 0.0 # -1, 0, or 1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.append(self.features[self.current_step], self.position)
        return obs.astype(np.float32)

    def step(self, action: int):
        # Map action 0->-1, 1->0, 2->1
        new_position = float(action) - 1.0
        
        reward = 0.0
        
        if self.current_step < self.n_steps - 1:
            price_today = self.prices[self.current_step]
            price_tomorrow = self.prices[self.current_step + 1]
            pct_change = (price_tomorrow - price_today) / price_today
            
            # The heart of the RL objective: capture alpha
            reward = new_position * pct_change
            
            # Transaction penalty prevents the agent from thrashing
            if new_position != self.position:
                reward -= 0.001
                
        self.position = new_position
        self.current_step += 1
        
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
