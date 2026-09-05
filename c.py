from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       

BASE_DIR = ""


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim + state_dim, latent_dim)
        self.posterior_log_std = nn.Linear(hidden_dim + state_dim, latent_dim)
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_log_std = nn.Linear(hidden_dim, latent_dim)
        
    def observe(self, latent, action, memory, next_state):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        posterior_input = torch.cat([memory, next_state], dim=-1)
        posterior_mean = self.posterior_mean(posterior_input)
        posterior_log_std = torch.clamp(self.posterior_log_std(posterior_input), -5, 2)
        posterior_std = torch.exp(posterior_log_std)
        posterior_dist = torch.distributions.Normal(posterior_mean, posterior_std)
        posterior_latent = posterior_dist.rsample()
        
        prior_mean = self.prior_mean(memory)
        prior_log_std = torch.clamp(self.prior_log_std(memory), -5, 2)
        prior_std = torch.exp(prior_log_std)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()
        
        return memory, posterior_dist, posterior_latent, prior_dist, prior_latent
        
    def imagine(self, latent, action, memory):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        prior_mean = self.prior_mean(memory)
        prior_log_std = torch.clamp(self.prior_log_std(memory), -5, 2)
        prior_std = torch.exp(prior_log_std)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()

        return memory, prior_latent
        
        
class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.recon = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        recon = self.recon(x)
        return recon
        
        
class RewardModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        reward = self.reward(x)
        return reward


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.rssm = RSSM(state_dim, action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
    
    def recon_loss(self, memory, posterior_latent, next_state):
        recon = self.decoder(memory, posterior_latent)
        recon_loss = F.mse_loss(next_state, recon)  
        return recon_loss
        
    def reward_loss(self, memory, posterior_latent, reward):
        pred_reward = self.reward_model(memory, posterior_latent)
        reward_loss = F.mse_loss(reward, pred_reward)
        return reward_loss
        
    def dist_loss(self, posterior_dist, prior_dist):
        dist_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        return dist_loss
        
    def loss(self, memory, posterior_latent, next_state, reward, posterior_dist, prior_dist):
        recon_loss = self.recon_loss(memory, posterior_latent, next_state)
        reward_loss = self.reward_loss(memory, posterior_latent, reward)
        dist_loss = self.dist_loss(posterior_dist, prior_dist)        
        total_loss = recon_loss + reward_loss + dist_loss
        
        return total_loss, recon_loss, reward_loss, dist_loss
        
    def forward(self, states, actions, rewards, next_states):
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
    
        B = states.shape[0]
        memory = torch.zeros(B, self.rssm.gru.hidden_size)
        latent = torch.zeros(B, self.rssm.prior_mean.out_features)
        
        memories = []
        posterior_latents = []
        posterior_dists = []
        prior_dists = []
        
        for t in range(states.shape[1]):
            memory, posterior_dist, posterior_latent, prior_dist, prior_latent = self.rssm.observe(latent, actions[:, t], memory, next_states[:, t])
            
            memories.append(memory)
            posterior_latents.append(posterior_latent)
            posterior_dists.append(posterior_dist)
            prior_dists.append(prior_dist)
            
            latent = posterior_latent
            
        memories = torch.stack(memories, dim=1)
        posterior_latents = torch.stack(posterior_latents, dim=1)
            
        return memories, posterior_latents, posterior_dists, prior_dists

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))       
        mean = self.mean(x)        
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()        
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
        
    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
        

class Critic(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        value = self.value(x)
        return value
        
        
class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic()
        self.world_model = WorldModel(state_dim, action_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        
    def update_world_model(self, state, action, next_state, reward):
        memories, posterior_latents, posterior_dists, prior_dists = self.world_model(state, action, reward, next_state)
        
        total_loss = 0
        recon_loss = 0
        reward_loss = 0
        dist_loss = 0
        
        for t in range(state.shape[1]):
            t_loss, rec_loss, rew_loss, d_loss = self.world_model.loss(memories[:, t, :], posterior_latents[:, t, :], next_state[:, t, :], reward[:, t, :], posterior_dists[t], prior_dists[t])    
            total_loss += t_loss
            recon_loss += rec_loss
            reward_loss += rew_loss
            dist_loss += d_loss
            
        total_loss /= state.shape[1]
        recon_loss /= state.shape[1]
        reward_loss /= state.shape[1]
        dist_loss /= state.shape[1]
        
        memory = memories[:, -1, :]
        latent = posterior_latents[:, -1, :]
        
        return total_loss, recon_loss, reward_loss, dist_loss, memory, latent            
        
    def imagine_with_AC(self, state, latent, memory, horizon=15):
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_next_states = []
        imagined_next_values = []
        
        for _ in range(horizon):
            action, _ = self.actor.sample(state)
            memory, latent = self.world_model.rssm.imagine(latent, action, memory)
            
            pred_next_state = self.world_model.decoder(memory, latent)
            pred_reward = self.world_model.reward_model(memory, latent)
            
            next_value = self.critic(memory, latent)
            
            imagined_states.append(state)
            imagined_actions.append(action)
            imagined_rewards.append(pred_reward)
            imagined_next_states.append(pred_next_state)
            imagined_next_values.append(next_value)

            state = pred_next_state
            
        imagined_states = torch.stack(imagined_states, dim=1)
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_next_states = torch.stack(imagined_next_states, dim=1)
        imagined_next_values = torch.stack(imagined_next_values, dim=1)
        
        return imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_next_values
        
    def compute_return(self, rewards, next_values, gamma=0.99, lambda_=0.95):
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        next_returns = next_values[:, -1, :]
        
        for t in reversed(range(H)):    
            next_returns = rewards[:, t, :] + gamma * ((1 - lambda_) * next_values[:, t, :] + lambda_ * next_returns)            
            returns[:, t, :] = next_returns
            
        return returns               
    
    def critic_loss(self, values, returns):
        critic_loss = F.mse_loss(values, returns)     
        return critic_loss
        
    def actor_loss(self, returns):          
        actor_loss = -returns.mean()       
        return actor_loss

    def update(self, batch):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])

        #==========================================
        
        world_model_loss, recon_loss, reward_loss, dist_loss, memory, latent = self.update_world_model(state, action, next_state, reward)
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False 
        for p in self.actor.parameters():
            p.requires_grad = False 
        
        imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_next_values = self.imagine_with_AC(state[:, -1, :], latent.detach(), memory.detach())
        with torch.no_grad():    
            returns = self.compute_return(imagined_rewards, imagined_next_values)
        
        critic_loss = self.critic_loss(imagined_next_values[:, :-1, :], returns[:, 1:, :])        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True 
        for p in self.actor.parameters():
            p.requires_grad = True 
        
        #==========================================
        
        for p in self.world_model.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False      
        
        imagined_states, imagined_actions, imagined_rewards, imagined_next_states, imagined_next_values = self.imagine_with_AC(state[:, -1, :], latent.detach(), memory.detach())
        returns = self.compute_return(imagined_rewards, imagined_next_values)
        
        actor_loss = self.actor_loss(returns)        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        for p in self.world_model.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True
        
        return {
            "world_model_loss": world_model_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),          
        }
        
        
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), batch_size=1, max_seq_len=50):
        self.max_size = max_size
        self.batch_size = batch_size
        self.current_seq_len = 1
        self.max_seq_len = max_seq_len
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)      
        self.agent_id = np.zeros(max_size, dtype=np.int64)
        self.episode_id = np.zeros(max_size, dtype=np.int64)

    def add(self, state, action, reward, next_state, done, agent_id, episode_id):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done       
        self.agent_id[self.ptr] = agent_id
        self.episode_id[self.ptr] = episode_id

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def update_seq_len(self):
        lengths = []

        for _ in range(self.batch_size):
            start = np.random.randint(0, self.size)

            agent_id = self.agent_id[start]
            episode_id = self.episode_id[start]

            indices = np.where((self.agent_id == agent_id) & (self.episode_id == episode_id))[0] 
            start_pos = np.where(indices == start)[0][0]

            length = len(indices) - start_pos
            lengths.append(length)

        max_length = max(lengths)
        if max_length >= self.current_seq_len:
            return min(self.current_seq_len + 1, self.max_seq_len)
        else:
            return self.current_seq_len
    
    def sample(self):
        if self.current_seq_len < self.max_seq_len:
            self.current_seq_len = self.update_seq_len()
            
        #print(self.current_seq_len)
        
        while True:    
            sequences = []
            attempts = 0
        
            while len(sequences) < self.batch_size:
                attempts += 1

                #if attempts % 10000 == 0:
                #    print(
                #        f"[ReplayBuffer] attempts={attempts}, "
                #        f"success={len(sequences)}/{self.batch_size}, "
                #        f"seq_len={self.current_seq_len}, "
                #        f"buffer_size={self.size}"
                #    )
                start = np.random.randint(0, self.size)
                if start + self.current_seq_len > self.size:
                    continue
            
                agent_id = self.agent_id[start]
                episode_id = self.episode_id[start]
            
                indices = np.where((self.agent_id[:self.size] == agent_id) & (self.episode_id[:self.size] == episode_id))[0]               
                start_pos = np.where(indices == start)[0][0]
                if start_pos + self.current_seq_len > len(indices):
                    self.current_seq_len = max(1, self.current_seq_len - 1)
                    break
                
                seq_indices = indices[start_pos:start_pos + self.current_seq_len]
                if np.any(self.done[seq_indices[:-1]]):
                    continue
                
                sequences.append(seq_indices)
                
            if len(sequences) == self.batch_size:
                break
            
        indices = np.stack(sequences)
        
        return {
            "state": self.state[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_state": self.next_state[indices],
            "done": self.done[indices],
        }
    

if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = Agent(state_dim, action_dim)
    #agent.actor.load_state_dict(torch.load(f"{BASE_DIR}/period_model.pth"))
    buffer = ReplayBuffer(state_dim, action_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    random_exploration_steps = 1000
    learning_starts = 500
    test_interval = 1000
    test_max_step = 1000
    
    episode_ids = {}
    next_episode_id = 0
    total_steps = 0
    update_count = 0
    save_idx = 0
    best_test_reward = -float('inf')
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)  
            
            if total_steps < random_exploration_steps:
                actions = np.random.uniform(low=-1.0, high=1.0, size=(len(agent_ids), action_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    actions, _ = agent.actor.sample(states_tensor)   
                actions = actions.cpu().numpy().astype(np.float32)
                
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in episode_ids:
                episode_ids[agent_id] = next_episode_id
                next_episode_id += 1               
            episode_id = episode_ids[agent_id]            
            state = states_tensor[i].cpu().numpy()
            action = actions[i]

            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = 1.0
                next_state = np.zeros_like(state)                
                episode_ids[agent_id] = next_episode_id
                next_episode_id += 1
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue
                
            buffer.add(state, action, reward, next_state, done, agent_id, episode_id)
            total_steps += 1
         
        if total_steps >= learning_starts:
             batch = buffer.sample()
             metrics = agent.update(batch) 
             update_count += 1           
             for k, v in metrics.items():
                 writer.add_scalar(f"Train/{k}", v, update_count)               
             
             if update_count % test_interval == 0:
                 print(f"Update Count {update_count}")
                 test_env.reset()
                 t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                 n_test_agents = len(t_decision_steps.agent_id)
                 test_rewards = np.zeros(n_test_agents)
                 test_episode_dones = np.zeros(n_test_agents, dtype=bool)
                 test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}
                 
                 test_max_step_count = 0
                 while not np.all(test_episode_dones) and test_max_step_count < test_max_step:
                     t_agent_ids = t_decision_steps.agent_id
                     
                     if len(t_agent_ids) > 0:
                         t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)                        
                         with torch.no_grad():
                             t_actions = agent.actor.deterministic(t_states_tensor)                    
                         t_actions = t_actions.cpu().numpy().astype(np.float32)
                         
                         for j, agent_id in enumerate(t_agent_ids):
                             idx = test_id_to_index[agent_id]
                             if test_episode_dones[idx]:
                                 t_actions[j] = np.zeros(action_dim)
                                
                         test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions))
                         
                     test_env.step()
                     test_max_step_count += 1
                     t_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)
                     
                     for j, agent_id in enumerate(t_terminal_steps.agent_id):
                         i = test_id_to_index[agent_id]
                         if not test_episode_dones[i]:
                             test_rewards[i] += t_terminal_steps.reward[j]
                             test_episode_dones[i] = True

                     for j, agent_id in enumerate(t_decision_steps.agent_id):
                         i = test_id_to_index[agent_id]
                         if not test_episode_dones[i]:
                             test_rewards[i] += t_decision_steps.reward[j]
                             
                 test_average_reward = np.mean(test_rewards)
                 writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
                 print(f"{test_average_reward:.4f}")
                 torch.save(agent.actor.state_dict(), f"{BASE_DIR}/period_model.pth")
                 #save_checkpoint(f"{BASE_DIR}/checkpoint.pth", agent, buffer)                    
                         
                 if test_average_reward > best_test_reward:
                     best_test_reward = test_average_reward
                     save_idx += 1
                     torch.save(agent.actor.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth") 
                     print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")
                     
