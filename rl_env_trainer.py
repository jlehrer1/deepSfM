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


class HumanoidEnvironmentTester:
    def __init__(self, meshes, mesh_dir="."):
        self.meshes = meshes
        self.mesh_dir = Path(mesh_dir)
        self.mesh_files = {}
        self.bounds = self._calculate_bounds()
        
    def _calculate_bounds(self):
        all_vertices = []
        for mesh in self.meshes.values():
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                all_vertices.append(vertices)
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            return {
                'min': all_vertices.min(axis=0),
                'max': all_vertices.max(axis=0)
            }
        return {'min': np.zeros(3), 'max': np.ones(3)}
    
    def save_meshes(self):
        """Save all meshes as OBJ files"""
        print("\nSaving meshes...")
        for name, mesh in self.meshes.items():
            filename = self.mesh_dir / f"{name}.obj"
            success = o3d.io.write_triangle_mesh(str(filename), mesh, write_ascii=True)
            if success:
                self.mesh_files[name] = filename.name
                print(f"  {name}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            else:
                print(f"  Failed to save {name}")
        return len(self.mesh_files) == len(self.meshes)
    
    def create_humanoid_mjcf_xml(self):
        """Create MuJoCo XML with humanoid agent"""
        
        ground_z = self.bounds['min'][2]
        agent_start_z = ground_z + 1.5  # Start humanoid higher
        
        xml = f"""<mujoco model="humanoid_sfm_environment">
    <compiler angle="degree" meshdir="{self.mesh_dir}" />
    <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="PGS"/>
    
    <default>
        <joint damping="1" stiffness="0" armature="0" limited="true" solreflimit="0.02" solimplimit="0.9"/>
        <geom condim="3" contype="1" conaffinity="1" friction="1 0.5 0.5" rgba="0.8 0.6 0.4 1"/>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
    </default>
    
    <asset>
"""
        
        # Add mesh assets
        for name, filename in self.mesh_files.items():
            xml += f'        <mesh name="{name}" file="{filename}"/>\n'
        
        xml += """    </asset>
    
    <worldbody>
"""
        
        # Add ground mesh
        if 'ground' in self.mesh_files:
            xml += """        <!-- Ground plane -->
        <geom name="ground" type="mesh" mesh="ground" pos="0 0 0" 
              rgba="0.4 0.3 0.2 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        # Add wall meshes
        for i in range(4):
            if f'wall_{i}' in self.mesh_files:
                xml += f"""        
        <!-- Wall {i} -->
        <geom name="wall_{i}" type="mesh" mesh="wall_{i}" pos="0 0 0" 
              rgba="0.8 0.8 0.9 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        # Add environment mesh
        if 'environment' in self.mesh_files:
            xml += """        
        <!-- Environment mesh (building) -->
        <geom name="environment" type="mesh" mesh="environment" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1"
              friction="1 0.5 0.5"/>
"""
        # Add humanoid agent
        xml += f"""
        <!-- Humanoid Agent -->
        <body name="torso" pos="0 0 {agent_start_z}">
            <freejoint name="root"/>
            <geom name="torso" type="capsule" size="0.07" fromto="0 0 -0.28 0 0 0.28" 
                  mass="10" rgba="0.8 0.6 0.4 1"/>
            
            <!-- Head -->
            <body name="head" pos="0 0 0.4">
                <geom name="head" type="sphere" size="0.09" mass="2" rgba="1 0.8 0.6 1"/>
            </body>
            
            <!-- Left Arm -->
            <body name="left_upper_arm" pos="0 0.18 0.2">
                <joint name="left_shoulder_x" type="hinge" axis="1 0 0" range="-85 60"/>
                <joint name="left_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="left_upper_arm" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" 
                      mass="2" rgba="0.8 0.6 0.4 1"/>
                
                <body name="left_lower_arm" pos="0 0 -0.3">
                    <joint name="left_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="left_lower_arm" type="capsule" size="0.031" fromto="0 0 0 0 0 -0.25" 
                          mass="1" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="left_hand" pos="0 0 -0.25">
                        <geom name="left_hand" type="sphere" size="0.04" mass="0.3" rgba="1 0.8 0.6 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Right Arm -->
            <body name="right_upper_arm" pos="0 -0.18 0.2">
                <joint name="right_shoulder_x" type="hinge" axis="1 0 0" range="-85 60"/>
                <joint name="right_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="right_upper_arm" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" 
                      mass="2" rgba="0.8 0.6 0.4 1"/>
                
                <body name="right_lower_arm" pos="0 0 -0.3">
                    <joint name="right_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="right_lower_arm" type="capsule" size="0.031" fromto="0 0 0 0 0 -0.25" 
                          mass="1" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="right_hand" pos="0 0 -0.25">
                        <geom name="right_hand" type="sphere" size="0.04" mass="0.3" rgba="1 0.8 0.6 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Left Leg -->
            <body name="left_thigh" pos="0 0.1 -0.28">
                <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="left_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
                <geom name="left_thigh" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.45" 
                      mass="4" rgba="0.8 0.6 0.4 1"/>
                
                <body name="left_shin" pos="0 0 -0.45">
                    <joint name="left_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="left_shin" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" 
                          mass="2.5" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="left_foot" pos="0 0 -0.4">
                        <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                        <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                        <geom name="left_foot" type="box" size="0.13 0.05 0.02" 
                              mass="1" rgba="0.9 0.7 0.5 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Right Leg -->
            <body name="right_thigh" pos="0 -0.1 -0.28">
                <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-35 60"/>
                <geom name="right_thigh" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.45" 
                      mass="4" rgba="0.8 0.6 0.4 1"/>
                
                <body name="right_shin" pos="0 0 -0.45">
                    <joint name="right_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="right_shin" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" 
                          mass="2.5" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="right_foot" pos="0 0 -0.4">
                        <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                        <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                        <geom name="right_foot" type="box" size="0.13 0.05 0.02" 
                              mass="1" rgba="0.9 0.7 0.5 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Arm motors -->
        <motor name="left_shoulder_x" joint="left_shoulder_x" gear="25"/>
        <motor name="left_shoulder_y" joint="left_shoulder_y" gear="25"/>
        <motor name="left_elbow" joint="left_elbow" gear="25"/>
        <motor name="right_shoulder_x" joint="right_shoulder_x" gear="25"/>
        <motor name="right_shoulder_y" joint="right_shoulder_y" gear="25"/>
        <motor name="right_elbow" joint="right_elbow" gear="25"/>
        
        <!-- Leg motors -->
        <motor name="left_hip_x" joint="left_hip_x" gear="100"/>
        <motor name="left_hip_y" joint="left_hip_y" gear="100"/>
        <motor name="left_hip_z" joint="left_hip_z" gear="100"/>
        <motor name="left_knee" joint="left_knee" gear="100"/>
        <motor name="left_ankle_y" joint="left_ankle_y" gear="50"/>
        <motor name="left_ankle_x" joint="left_ankle_x" gear="50"/>
        
        <motor name="right_hip_x" joint="right_hip_x" gear="100"/>
        <motor name="right_hip_y" joint="right_hip_y" gear="100"/>
        <motor name="right_hip_z" joint="right_hip_z" gear="100"/>
        <motor name="right_knee" joint="right_knee" gear="100"/>
        <motor name="right_ankle_y" joint="right_ankle_y" gear="50"/>
        <motor name="right_ankle_x" joint="right_ankle_x" gear="50"/>
    </actuator>
</mujoco>
"""
        
        return xml


class HumanoidWalkingEnv:
    """MuJoCo Humanoid Walking Environment"""
    
    def __init__(self, model_xml):
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.action_dim = self.model.nu  # Number of actuators
        self.obs_dim = self._get_observation().shape[0]
        
        # Episode tracking
        self.episode_length = 1000
        self.step_count = 0
        
        # Reference positions for initial pose
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
    def _get_observation(self):
        """Get current observation state"""
        # Get positions and velocities
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Get center of mass
        com_pos = self.data.subtree_com[1].copy()  # torso com
        com_vel = self.data.cvel[1].copy()[:3]     # linear velocity
        
        # Get torso orientation
        torso_quat = qpos[3:7]
        
        # Get foot contact forces
        left_foot_contact = self._get_contact_force("left_foot")
        right_foot_contact = self._get_contact_force("right_foot")
        
        # Combine observations
        obs = np.concatenate([
            qpos[7:],  # joint positions (excluding root translation/rotation)
            qvel,      # all velocities
            com_pos,   # center of mass position
            com_vel,   # center of mass velocity
            torso_quat, # torso orientation
            [left_foot_contact, right_foot_contact]  # foot contacts
        ])
        
        return obs
    
    def _get_contact_force(self, geom_name):
        """Get contact force magnitude for a geom"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        contact_force = 0.0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                # Get contact force
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force += np.linalg.norm(c_array[:3])
        
        return min(contact_force / 1000.0, 1.0)  # Normalize
    
    def reset(self):
        """Reset environment to initial state"""
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        
        # Add small random perturbation
        self.data.qpos[7:] += np.random.normal(0, 0.02, size=len(self.data.qpos[7:]))
        self.data.qvel[:] += np.random.normal(0, 0.01, size=len(self.data.qvel))
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Take environment step"""
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._is_terminal()
        
        self.step_count += 1
        if self.step_count >= self.episode_length:
            done = True
        
        return obs, reward, done, {}
    
    def _calculate_reward(self):
        """Calculate walking reward"""
        # Forward velocity reward
        forward_vel = self.data.cvel[1][0]  # torso forward velocity
        forward_reward = min(forward_vel, 2.0)  # Cap at 2 m/s
        
        # Height penalty (want to stay upright)
        torso_height = self.data.qpos[2]
        height_reward = -abs(torso_height - 1.3)  # Target height around 1.3m
        
        # Orientation penalty (stay upright)
        torso_quat = self.data.qpos[3:7]
        up_vec = np.array([0, 0, 1])
        torso_up = self._quat_to_up_vector(torso_quat)
        upright_reward = np.dot(torso_up, up_vec) - 1.0
        
        # Energy penalty (smooth movement)
        action_penalty = -0.01 * np.sum(self.data.ctrl ** 2)
        
        # Contact reward (encourage ground contact)
        left_contact = self._get_contact_force("left_foot")
        right_contact = self._get_contact_force("right_foot")
        contact_reward = 0.1 * (left_contact + right_contact)
        
        # Total reward
        total_reward = (forward_reward + 
                       0.5 * height_reward + 
                       0.5 * upright_reward + 
                       action_penalty + 
                       contact_reward)
        
        return total_reward
    
    def _quat_to_up_vector(self, quat):
        """Convert quaternion to up vector"""
        w, x, y, z = quat
        up = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        return up / np.linalg.norm(up)
    
    def _is_terminal(self):
        """Check if episode should terminate"""
        # Terminate if fallen (torso too low)
        if self.data.qpos[2] < 0.5:
            return True
        
        # Terminate if torso tilted too much
        torso_quat = self.data.qpos[3:7]
        up_vec = np.array([0, 0, 1])
        torso_up = self._quat_to_up_vector(torso_quat)
        if np.dot(torso_up, up_vec) < 0.3:  # More than ~70 degrees tilt
            return True
        
        return False


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
    
