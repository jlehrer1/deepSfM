"""
MuJoCo Humanoid Walking Training for SfM Reconstructions

Trains a humanoid agent to walk in environments created from Structure from Motion data.
Uses PPO (Proximal Policy Optimization) with actor-critic networks.
"""

import numpy as np
import open3d as o3d
import mujoco
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    from deepsfm.mesh_creator import create_rl_environment
except ImportError:
    print("Error: Could not import mesh_creator module")
    print("Make sure the RLMeshCreator class is saved as 'mesh_creator.py'")

class ActorCritic(nn.Module):
    """Actor-Critic network for humanoid control"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        # Shared features
        x = F.tanh(self.shared_fc1(state))
        x = F.tanh(self.shared_fc2(x))
        
        actor_x = F.tanh(self.actor_fc(x))
        action_mean = self.actor_mean(actor_x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        critic_x = F.tanh(self.critic_fc(x))
        value = self.critic_value(critic_x)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        action_logprob = torch.distributions.Normal(action_mean, action_std).log_prob(action)
        action_logprob = action_logprob.sum(dim=-1, keepdim=True)
        
        return action, action_logprob, value


class PPOTrainer:
    """PPO Trainer for humanoid walking"""
    
    def __init__(self, env, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize networks
        self.actor_critic = ActorCritic(env.obs_dim, env.action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def collect_rollout(self, rollout_length=2048):
        """Collect rollout data"""
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        state = self.env.reset()
        
        for _ in range(rollout_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action(state_tensor)
            
            action_np = action.squeeze().numpy()
            next_state, reward, done, _ = self.env.step(action_np)
            
            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            
            if done:
                state = self.env.reset()
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'rewards': np.array(rewards),
            'dones': np.array(dones)
        }
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, rollout_data, num_epochs=10, batch_size=64):
        """Update policy using PPO"""
        states = torch.FloatTensor(rollout_data['states'])
        actions = torch.FloatTensor(rollout_data['actions'])
        old_log_probs = torch.FloatTensor(rollout_data['log_probs'])
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            rollout_data['rewards'], 
            rollout_data['values'], 
            rollout_data['dones']
        )
        
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_mean, action_std, values = self.actor_critic(batch_states)
                
                # Calculate new log probabilities
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Calculate ratios
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, total_timesteps=1000000, rollout_length=2048, log_interval=10, 
              save_interval=100, visualize_interval=50):
        """Main training loop"""
        timesteps = 0
        episode = 0
        
        print("Starting PPO training...")
        print(f"Observation dim: {self.env.obs_dim}, Action dim: {self.env.action_dim}")
        
        for timestep in tqdm(range(total_timesteps)):
            if timesteps >= total_timesteps:
                break 

            # Collect rollout
            rollout_data = self.collect_rollout(rollout_length)
            timesteps += rollout_length
            
            # Update policy
            self.update_policy(rollout_data)
            
            # Calculate episode stats
            episode_reward = np.sum(rollout_data['rewards'])
            episode_length = len(rollout_data['rewards'])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            episode += 1
            
            # Logging
            if episode % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                print(f"Episode {episode}, Timesteps {timesteps}")
                print(f"  Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
            
            # Visualization
            if episode % visualize_interval == 0:
                print(f"\nVisualizing episode {episode}...")
                self.visualize_episode()
            
            # Save model
            if episode % save_interval == 0:
                torch.save(self.actor_critic.state_dict(), f'humanoid_walking_{episode}.pth')
                print(f"Model saved at episode {episode}")
    
    def visualize_episode(self, max_steps=1000):
        """Run and visualize a single episode"""
        try:
            if hasattr(mujoco, 'viewer'):
                viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
                
                state = self.env.reset()
                total_reward = 0
                steps = 0
                
                while viewer.is_running() and steps < max_steps:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        action, _, _ = self.actor_critic.get_action(state_tensor, deterministic=True)
                    
                    action_np = action.squeeze().numpy()
                    state, reward, done, _ = self.env.step(action_np)
                    
                    total_reward += reward
                    steps += 1
                    
                    viewer.sync()
                    time.sleep(0.01)
                    
                    if done:
                        print(f"Episode finished: {steps} steps, reward: {total_reward:.2f}")
                        break
                
                print(f"Visualization complete: {steps} steps, total reward: {total_reward:.2f}")
            else:
                print("Interactive viewer not available in this MuJoCo version")
                
        except Exception as e:
            print(f"Visualization error: {e}")


def train_humanoid_walking(data_path='camera_view_ground_env.npz', 
                          visualize_meshes=False,
                          training_timesteps=1000000):
    """
    Complete pipeline to train humanoid walking in SfM environment.
    
    Args:
        data_path: Path to normalized SfM data (.npz file)
        visualize_meshes: Show meshes in Open3D before training
        training_timesteps: Total training timesteps
    """
    data = np.load(data_path)
    normalized_data = {
        'points': data['points'],
        'colors': data['colors'],
        'camera_positions': data.get('camera_positions', []),
        'camera_directions': data.get('camera_directions', [])
    }
    
    env_data = create_rl_environment(
        normalized_data, 
        add_walls=True, 
        simplify=True, 
        visualize=visualize_meshes
    )
    tester = HumanoidEnvironmentTester(env_data['meshes'])
    
    # Save meshes
    if not tester.save_meshes():
        print("Failed to save meshes!")
        return False
    
    # Create and save XML
    xml = tester.create_humanoid_mjcf_xml()
    xml_path = tester.mesh_dir / "humanoid_environment.xml"
    with open(xml_path, 'w') as f:
        f.write(xml)
    print(f"Saved MuJoCo XML: {xml_path}")
    
    env = HumanoidWalkingEnv(xml)
    trainer = PPOTrainer(env)
    trainer.train(total_timesteps=training_timesteps)
    return True


if __name__ == "__main__":
    success = train_humanoid_walking(
        data_path='camera_view_ground_env.npz',
        visualize_meshes=False,
        training_timesteps=500000
    )
    
