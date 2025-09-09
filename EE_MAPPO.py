import os
import shutil
from time import sleep
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from gymnasium_robotics import mamujoco_v1
from envs.Continuous2D import Continuous2DEnv
import math
import random

RND_FEATURE_DIM = 128
RND_LR = 1e-5
RND_REWARD_SCALE = 10
RND_UPDATE_FREQ = 1

ENV_CRITIC_LR = 1e-4
RND_CRITIC_LR = 1e-4
CRITIC_WEIGHT_ENV = 0.7
CRITIC_WEIGHT_RND = 0.3

result_base_dir = "results"
env_name = "EE_MAPPO"
result_dir = os.path.join(result_base_dir, env_name)

logs_dir = os.path.join(result_dir, "logs")
plots_dir = os.path.join(result_dir, "plots")
weights_dir = os.path.join(result_dir, "weights")

for directory in [logs_dir, plots_dir, weights_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

log_file = os.path.join(logs_dir, "training_log.txt")

if os.path.exists(log_file):
    open(log_file, "w").close()

def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def plot_all_metrics(metrics_dict, episode, death_rates=None, finished_rates=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics of {env_name} (Up to Episode {episode})', fontsize=16)
    
    axes = axes.flatten()
    
    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    
    window_size = min(5, len(x_values)) if len(x_values) > 0 else 1
    
    current_metric_index = 0
    for metric_name, values in metrics_dict.items():
        if current_metric_index >= 5:
            break
            
        ax = axes[current_metric_index]
        values_array = np.array(values)
        
        if len(values) > window_size:
            smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            
            std_values = []
            for j in range(len(values) - window_size + 1):
                std_values.append(np.std(values_array[j:j+window_size]))
            std_values = np.array(std_values)
            
            smoothed_x = x_values[window_size-1:]
            
            ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            
            ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, 
                           alpha=0.2, label='±1 StdDev')
        else:
            ax.plot(x_values, values, 'o-', label='Data')
        
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        current_metric_index += 1
    
    if death_rates is not None and finished_rates is not None:
        ax = axes[5]
        
        death_x_values = [50 * (i + 1) for i in range(len(death_rates))]
        finished_x_values = [50 * (i + 1) for i in range(len(finished_rates))]
        
        if len(death_rates) > window_size:
            death_smoothed = np.convolve(np.array(death_rates), np.ones(window_size)/window_size, mode='valid')
            death_smoothed_x = death_x_values[window_size-1:]
            ax.plot(death_smoothed_x, death_smoothed, '-', linewidth=2, color='red', label='Death Rate (Smoothed)')
            ax.scatter(death_x_values, death_rates, alpha=0.3, color='red', label='Death Rate (Original)')
        else:
            ax.plot(death_x_values, death_rates, 'o-', color='red', label='Death Rate')
            
        if len(finished_rates) > window_size:
            finished_smoothed = np.convolve(np.array(finished_rates), np.ones(window_size)/window_size, mode='valid')
            finished_smoothed_x = finished_x_values[window_size-1:]
            ax.plot(finished_smoothed_x, finished_smoothed, '-', linewidth=2, color='green', label='Success Rate (Smoothed)')
            ax.scatter(finished_x_values, finished_rates, alpha=0.3, color='green', label='Success Rate (Original)')
        else:
            ax.plot(finished_x_values, finished_rates, 'o-', color='green', label='Success Rate')
        
        ax.set_title('Death Rate vs Success Rate')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, f'EE_MAPPO_training_metrics.png'))
    plt.close(fig)

def compute_entropy(mean, log_std):
    std = log_std.exp()
    entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
    return entropy.mean().item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.mean(x)
        mean = torch.tanh(mean)*2
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -0.9, 2)
        return mean, log_std

class RNDTargetNet(torch.nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(RNDTargetNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc3 = torch.nn.Linear(feature_dim, feature_dim)
        
        self._freeze_parameters()
    
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNDPredictNet(torch.nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(RNDPredictNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc3 = torch.nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EnvValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(EnvValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, team_size)

    def forward(self, x):
        x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return self.fc4(x)  # [batch, team_size]

class RNDValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(RNDValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, team_size)

    def forward(self, x):
        x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return self.fc4(x)  # [batch, team_size]


class EE_MAPPO:
    def __init__(self, team_size, state_dim, hidden_dim, action_dim,
                 actor_lr, env_critic_lr, rnd_critic_lr, lmbda, eps, gamma, device):
        self.team_size = team_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device

        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim).to(device)
                       for _ in range(team_size)]

        self.env_critic = EnvValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        self.rnd_critic = RNDValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), actor_lr) 
                                 for actor in self.actors]
        self.env_critic_optimizer = torch.optim.Adam(self.env_critic.parameters(), env_critic_lr)
        self.rnd_critic_optimizer = torch.optim.Adam(self.rnd_critic.parameters(), rnd_critic_lr)

        self.rnd_target_nets = [RNDTargetNet(state_dim, RND_FEATURE_DIM).to(device) 
                               for _ in range(team_size)]
        self.rnd_predict_nets = [RNDPredictNet(state_dim, RND_FEATURE_DIM).to(device) 
                                for _ in range(team_size)]
        self.rnd_optimizers = [torch.optim.Adam(predict_net.parameters(), RND_LR) 
                              for predict_net in self.rnd_predict_nets]
        
        self.rnd_rewards_history = [[] for _ in range(team_size)]
        self.rnd_losses_history = [[] for _ in range(team_size)]

    def compute_rnd_reward(self, states):
        rnd_rewards = []
        for i in range(self.team_size):
            state_tensor = torch.tensor(states[i], dtype=torch.float).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                target_features = self.rnd_target_nets[i](state_tensor)
            predict_features = self.rnd_predict_nets[i](state_tensor)
            
            rnd_error = F.mse_loss(predict_features, target_features, reduction='none').mean(dim=1)
            rnd_reward = rnd_error.detach().cpu().numpy().flatten()[0]
            
            rnd_rewards.append(rnd_reward)
            self.rnd_rewards_history[i].append(rnd_reward)
        
        return rnd_rewards

    def update_rnd(self, states):
        rnd_losses = []
        for i in range(self.team_size):
            state_tensor = torch.tensor(states[i], dtype=torch.float).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                target_features = self.rnd_target_nets[i](state_tensor)
            predict_features = self.rnd_predict_nets[i](state_tensor)
            
            rnd_loss = F.mse_loss(predict_features, target_features)
            
            self.rnd_optimizers[i].zero_grad()
            rnd_loss.backward()
            self.rnd_optimizers[i].step()
            
            rnd_losses.append(rnd_loss.item())
            self.rnd_losses_history[i].append(rnd_loss.item())
        
        return rnd_losses

    def update_rnd_single_agent(self, agent_id, states):
        with torch.no_grad():
            target_features = self.rnd_target_nets[agent_id](states)
        predict_features = self.rnd_predict_nets[agent_id](states)
        
        rnd_loss = F.mse_loss(predict_features, target_features)
        
        self.rnd_optimizers[agent_id].zero_grad()
        rnd_loss.backward()
        self.rnd_optimizers[agent_id].step()
        
        return rnd_loss.item()

    def save_model(self, path=None):
        if path is None:
            path = weights_dir
        if not os.path.exists(path):
            os.makedirs(path)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f"actor_{i}.pth"))
        torch.save(self.env_critic.state_dict(), os.path.join(path, "env_critic.pth"))
        torch.save(self.rnd_critic.state_dict(), os.path.join(path, "rnd_critic.pth"))
        
        for i in range(self.team_size):
            torch.save(self.rnd_predict_nets[i].state_dict(), 
                      os.path.join(path, f"rnd_predict_{i}.pth"))
            torch.save(self.rnd_target_nets[i].state_dict(), 
                      os.path.join(path, f"rnd_target_{i}.pth"))

    def load_model(self, path=None):
        if path is None:
            path = weights_dir
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path))
        env_critic_path = os.path.join(path, "env_critic.pth")
        if os.path.exists(env_critic_path):
            self.env_critic.load_state_dict(torch.load(env_critic_path))
        rnd_critic_path = os.path.join(path, "rnd_critic.pth")
        if os.path.exists(rnd_critic_path):
            self.rnd_critic.load_state_dict(torch.load(rnd_critic_path))
        
        for i in range(self.team_size):
            rnd_predict_path = os.path.join(path, f"rnd_predict_{i}.pth")
            rnd_target_path = os.path.join(path, f"rnd_target_{i}.pth")
            if os.path.exists(rnd_predict_path):
                self.rnd_predict_nets[i].load_state_dict(torch.load(rnd_predict_path))
            if os.path.exists(rnd_target_path):
                self.rnd_target_nets[i].load_state_dict(torch.load(rnd_target_path))

    def take_action(self, state_per_agent):
        actions = []
        action_probs = []
        
        rnd_rewards = self.compute_rnd_reward(state_per_agent)
        
        for i, actor in enumerate(self.actors):
            s = torch.tensor(np.array([state_per_agent[i]]), dtype=torch.float).to(self.device)
            mean, log_std = actor(s)
            std = log_std.exp()
            
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
           
            action_prob = normal.log_prob(action).sum(dim=-1).exp()
            
            actions.append(action.cpu().detach().numpy().flatten())
            action_probs.append(action_prob.cpu().detach().numpy())
        
        return actions, action_probs, rnd_rewards

    def update(self, transition_dicts, state_dim):
        T = len(transition_dicts[0]['states'])
        states_all = []
        next_states_all = []
        for t in range(T):
            concat_state = []
            concat_next_state = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]['states'][t])
                concat_next_state.append(transition_dicts[i]['next_states'][t])
            states_all.append(np.concatenate(concat_state))
            next_states_all.append(np.concatenate(concat_next_state))

        states_all = torch.tensor(states_all, dtype=torch.float).to(self.device)  # [T, team_size*state_dim]
        next_states_all = torch.tensor(next_states_all, dtype=torch.float).to(self.device) # [T, team_size*state_dim]

        rewards_all = torch.tensor([[transition_dicts[i]['rewards'][t] for i in range(self.team_size)] 
                                     for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size]
        dones_all = torch.tensor([[transition_dicts[i]['dones'][t] for i in range(self.team_size)] 
                                   for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size] 

        rnd_rewards_all = []
        for t in range(T):
            rnd_rewards_t = []
            for i in range(self.team_size):
                state_tensor = torch.tensor(transition_dicts[i]['states'][t], dtype=torch.float).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    target_features = self.rnd_target_nets[i](state_tensor)
                predict_features = self.rnd_predict_nets[i](state_tensor)
                rnd_error = F.mse_loss(predict_features, target_features, reduction='none').mean(dim=1)
                rnd_reward = rnd_error.detach().cpu().numpy().flatten()[0]
                rnd_rewards_t.append(rnd_reward)
            rnd_rewards_all.append(rnd_rewards_t)
        rnd_rewards_all = torch.tensor(rnd_rewards_all, dtype=torch.float).to(self.device) # [T, team_size]

        env_values = self.env_critic(states_all) # [T, team_size]    
        env_next_values = self.env_critic(next_states_all) # [T, team_size]
        rnd_values = self.rnd_critic(states_all) # [T, team_size]    
        rnd_next_values = self.rnd_critic(next_states_all) # [T, team_size]

        env_td_target = rewards_all + self.gamma * env_next_values * (1 - dones_all) # [T, team_size]
        rnd_td_target = rnd_rewards_all + self.gamma * rnd_next_values * (1 - dones_all) # [T, team_size]
        
        env_td_delta = env_td_target - env_values # [T, team_size]
        rnd_td_delta = rnd_td_target - rnd_values # [T, team_size]

        advantages = []
        for i in range(self.team_size):
            env_adv_i = compute_advantage(self.gamma, self.lmbda, env_td_delta[:, i])
            rnd_adv_i = compute_advantage(self.gamma, self.lmbda, rnd_td_delta[:, i])
            
            combined_adv = CRITIC_WEIGHT_ENV * env_adv_i + CRITIC_WEIGHT_RND * rnd_adv_i
            advantages.append(combined_adv.to(self.device))  # [T]

        env_critic_loss = F.mse_loss(env_values, env_td_target.detach())
        rnd_critic_loss = F.mse_loss(rnd_values, rnd_td_target.detach())
        
        self.env_critic_optimizer.zero_grad()
        env_critic_loss.backward()
        self.env_critic_optimizer.step()
        
        self.rnd_critic_optimizer.zero_grad()
        rnd_critic_loss.backward()
        self.rnd_critic_optimizer.step()

        action_losses = []
        entropies = []
        rnd_losses = []

        for i in range(self.team_size):
            states = torch.tensor(transition_dicts[i]['states'], dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dicts[i]['actions'], dtype=torch.float).to(self.device)
            old_probs = torch.tensor(transition_dicts[i]['action_probs'], dtype=torch.float).to(self.device)

            mean, log_std = self.actors[i](states)
            std = log_std.exp()
            normal_dist = torch.distributions.Normal(mean, std)
            
            log_probs = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
            old_log_probs = torch.log(old_probs).detach()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages[i].unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[i].unsqueeze(-1)
            entropy_val = torch.mean(normal_dist.entropy()).item()

            action_loss = torch.mean(-torch.min(surr1, surr2)) - 10* entropy_val
            
            self.actor_optimizers[i].zero_grad()
            action_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)
            
            if 'rnd_states' in transition_dicts[i]:
                rnd_states = torch.tensor(transition_dicts[i]['rnd_states'], dtype=torch.float).to(self.device)
                rnd_loss = self.update_rnd_single_agent(i, rnd_states)
                rnd_losses.append(rnd_loss)

        return np.mean(action_losses), env_critic_loss.item(), rnd_critic_loss.item(), np.mean(entropies), np.mean(rnd_losses) if rnd_losses else 0.0


actor_lr = 1e-4
env_critic_lr = 3e-4
rnd_critic_lr = 1e-4
total_episodes = 5000
hidden_dim = 128
gamma = 0.99
lmbda = 0.97
eps = 0.3
team_size = 2
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
best_finished_rate = 0.0

trajectory_save_interval = 100
env = Continuous2DEnv(num_agents=team_size, num_obstacles=0,use_radar=False)
states, info = env.reset()
agent_order = list(env.agents)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ee_mappo = EE_MAPPO(team_size, state_dim, hidden_dim, action_dim, actor_lr, env_critic_lr, rnd_critic_lr, lmbda, eps, gamma, device)

total_rewards_per_episode = []
episode_lengths = []
policy_losses = []
env_value_losses = []
rnd_value_losses = []
entropies = []
rnd_losses = []
episode_deaths = []
finished_episodes = []

avg_total_rewards_per_50 = []
avg_episode_length_per_50 = []
avg_policy_loss_per_50 = []
avg_env_value_loss_per_50 = []
avg_rnd_value_loss_per_50 = []
avg_entropy_per_50 = []
avg_rnd_loss_per_50 = []
avg_death_per_50 = []
avg_finished_per_50 = []

trajectories_dir = os.path.join(result_dir, "trajectories")
if not os.path.exists(trajectories_dir):
    os.makedirs(trajectories_dir)

with tqdm(total=total_episodes, desc="Training") as pbar:
    for episode in range(1, total_episodes + 1):
        buffers = [{
            'states': [], 
            'actions': [], 
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': [],
            'rnd_states': []
        } for _ in range(team_size)]
        
        current_states, _ = env.reset()
        
        agent_order = list(env.agents)
        terminal = False
        episode_reward = 0.0
        episode_death = 0.0
        episode_finished = 0.0
        steps = 0
        
        trajectory_data = {
            'episode': episode,
            'agent_positions': [],
            'goal_positions': [],
            'obstacle_positions': env.obstacles.copy(),
            'rewards': [],
            'finished': [],
            'deaths': [],
            'step_count': 0
        }

        while not terminal:
            steps += 1
            state_list = [current_states[agent] for agent in agent_order]
            actions, prob_dists, rnd_rewards = ee_mappo.take_action(state_list)
            
            
            action_dict = {agent_order[i]: actions[i] for i in range(team_size)}
            
            next_states, rewards, finished, truncations, death = env.step(action_dict)
            dones = {agent: finished[agent] or truncations[agent] or death[agent] for agent in agent_order}
            truncations = {agent: truncations[agent]  for agent in agent_order}
            
            rnd_total_reward = sum(rnd_rewards)
            step_reward = sum(rewards.values()) + RND_REWARD_SCALE * rnd_total_reward
            episode_reward += step_reward

            step_death = sum(death.values())
            episode_death += step_death

            step_finished = sum(finished.values())
            episode_finished += step_finished

            trajectory_data['agent_positions'].append(env.agent_pos.copy())
            trajectory_data['goal_positions'].append(env.goal_pos.copy())
            trajectory_data['rewards'].append(rewards.copy())
            trajectory_data['finished'].append(finished.copy())
            trajectory_data['deaths'].append(death.copy())
            trajectory_data['step_count'] = steps

            for i, agent in enumerate(agent_order):
                buffers[i]['states'].append(np.array(current_states[agent]))
                buffers[i]['actions'].append(actions[i])
                buffers[i]['next_states'].append(np.array(next_states[agent]))
                buffers[i]['rewards'].append(rewards[agent])
                buffers[i]['dones'].append(float(dones[agent]))
                buffers[i]['action_probs'].append(prob_dists[i])
                buffers[i]['rnd_states'].append(np.array(current_states[agent]))

            current_states = next_states
            terminal = all(dones.values())

        a_loss, env_c_loss, rnd_c_loss, ent, rnd_loss = ee_mappo.update(buffers, state_dim)

        total_rewards_per_episode.append(episode_reward)
        episode_lengths.append(steps)
        policy_losses.append(a_loss)
        env_value_losses.append(env_c_loss)
        rnd_value_losses.append(rnd_c_loss)
        entropies.append(ent)
        rnd_losses.append(rnd_loss)
        episode_deaths.append(episode_death)
        finished_episodes.append(episode_finished)
        
        should_save_trajectory = (episode % trajectory_save_interval == 0) 
        
        if should_save_trajectory:
            
            trajectory_file = os.path.join(trajectories_dir, f"trajectory_episode_{episode}_reward_{episode_reward:.3f}.npz")
            
            rewards_array = np.array([[trajectory_data['rewards'][t][f'agent_{i}'] 
                                     for i in range(team_size)] for t in range(len(trajectory_data['rewards']))])
            finished_array = np.array([[trajectory_data['finished'][t][f'agent_{i}'] 
                                      for i in range(team_size)] for t in range(len(trajectory_data['finished']))])
            deaths_array = np.array([[trajectory_data['deaths'][t][f'agent_{i}'] 
                                    for i in range(team_size)] for t in range(len(trajectory_data['deaths']))])
            
            np.savez_compressed(
                trajectory_file,
                agent_positions=np.array(trajectory_data['agent_positions']),
                goal_positions=np.array(trajectory_data['goal_positions']),
                obstacle_positions=trajectory_data['obstacle_positions'],
                rewards=rewards_array,
                finished=finished_array,
                deaths=deaths_array,
                step_count=trajectory_data['step_count'],
                episode=episode,
                total_reward=episode_reward
            )
            

            log_message(f"Trajectory saved: {trajectory_file} (Episode {episode}, Reward: {episode_reward:.3f})")

        #if episode % 500 == 0:
        
        if len(avg_finished_per_50) > 0 and avg_finished_per_50[-1] > 80 and avg_finished_per_50[-1] > best_finished_rate:
            ee_mappo.save_model()
            log_message(f"Model saved at episode {episode}")
            best_finished_rate = avg_finished_per_50[-1]

        if episode % 50 == 0:
            avg_reward_50 = np.mean(total_rewards_per_episode[-50:])
            avg_length_50 = np.mean(episode_lengths[-50:])
            avg_policy_loss_50 = np.mean(policy_losses[-50:])
            avg_env_value_loss_50 = np.mean(env_value_losses[-50:])
            avg_rnd_value_loss_50 = np.mean(rnd_value_losses[-50:])
            avg_entropy_50 = np.mean(entropies[-50:])
            avg_rnd_loss_50 = np.mean(rnd_losses[-50:])
            avg_death_50_rate = np.sum(episode_deaths[-50:])/(50*team_size)*100
            avg_finished_50 = np.sum(finished_episodes[-50:])/(50*team_size)*100

            avg_total_rewards_per_50.append(avg_reward_50)
            avg_episode_length_per_50.append(avg_length_50)
            avg_policy_loss_per_50.append(avg_policy_loss_50)
            avg_env_value_loss_per_50.append(avg_env_value_loss_50)
            avg_rnd_value_loss_per_50.append(avg_rnd_value_loss_50)
            avg_entropy_per_50.append(avg_entropy_50)
            avg_rnd_loss_per_50.append(avg_rnd_loss_50)
            avg_death_per_50.append(avg_death_50_rate)
            avg_finished_per_50.append(avg_finished_50)


            
            log_message(f"Episode {episode}: "
                        f"AvgTotalReward(last50)={avg_reward_50:.3f}, "
                        f"AvgEpisodeLength(last50)={avg_length_50:.3f}, "
                        f"AvgPolicyLoss(last50)={avg_policy_loss_50:.3f}, "
                        f"AvgEnvValueLoss(last50)={avg_env_value_loss_50:.3f}, "
                        f"AvgRNDValueLoss(last50)={avg_rnd_value_loss_50:.3f}, "
                        f"AvgEntropy(last50)={avg_entropy_50:.3f}, "
                        f"AvgRNDLoss(last50)={avg_rnd_loss_50:.3f}, "
                        f"AvgDeath(last50)={avg_death_50_rate:.3f}, "
                        f"AvgFinished(last50)={avg_finished_50:.3f}"
                        )
                
            metrics_dict = {
                "Average_Total_Reward": avg_total_rewards_per_50,
                "Average_Episode_Length": avg_episode_length_per_50,
                "Average_Policy_Loss": avg_policy_loss_per_50,
                "Average_Env_Value_Loss": avg_env_value_loss_per_50,
                "Average_RND_Value_Loss": avg_rnd_value_loss_per_50,
                "Average_Entropy": avg_entropy_per_50,
                "Average_RND_Loss": avg_rnd_loss_per_50
            }
                
            plot_all_metrics(metrics_dict, episode, avg_death_per_50, avg_finished_per_50)
            

        pbar.update(1)
