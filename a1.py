from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

BASE_DIR = ""


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.embed = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, state):
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        embed = self.embed(x)
        return embed
        

class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim=64, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)        
        self.posterior_mean = nn.Linear(hidden_dim + embed_dim, latent_dim)
        self.posterior_std = nn.Linear(hidden_dim + embed_dim, latent_dim)        
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)         
        
    def observe(self, latent, action, memory, embed):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        posterior_input = torch.cat([memory, embed], dim=-1)
        posterior_mean = self.posterior_mean(posterior_input)
        posterior_std = F.softplus(self.posterior_std(posterior_input)) + 1e-4
        posterior_dist = torch.distributions.Normal(posterior_mean, posterior_std)
        posterior_latent = posterior_dist.rsample()
        
        prior_mean = self.prior_mean(memory)
        prior_std = F.softplus(self.prior_std(memory)) + 1e-4
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()
        
        return memory, posterior_dist, posterior_latent, prior_dist, prior_latent
        
    def imagine(self, latent, action, memory):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        prior_mean = self.prior_mean(memory)
        prior_std = F.softplus(self.prior_std(memory)) + 1e-4
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_latent = prior_dist.rsample()
        
        return memory, prior_latent

        
class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, latent_dim=64):
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
    def __init__(self, hidden_dim=256, latent_dim=64):
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
    def __init__(self, state_dim, action_dim, memory_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
    def recon_loss(self, memory, posterior_latent, state):
        recon = self.decoder(memory, posterior_latent)
        recon_loss = F.mse_loss(state, recon)  
        return recon_loss
        
    def reward_loss(self, memory, posterior_latent, reward):
        pred_reward = self.reward_model(memory, posterior_latent)
        reward_loss = F.mse_loss(reward, pred_reward)
        return reward_loss
        
    def dist_loss(self, posterior_dist, prior_dist):
        dist_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        return dist_loss  
    
    def update(self, episodes, starts, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        sequence_length = states.shape[1]
        
        total_recon_loss = 0.0
        total_reward_loss = 0.0
        total_dist_loss = 0.0
        
        memories = []
        latents = []
        
        with torch.no_grad():
            for i in range(batch_size):
                initial_memory = torch.zeros(1, self.memory_dim)
                initial_latent = torch.zeros(1, self.latent_dim)
                initial_action = torch.zeros(1, self.action_dim)
                
                state = episodes[i][0]["state"].unsqueeze(0)
                embed = self.encoder(state)
                memory, _, latent, _, _ = self.rssm.observe(initial_latent, initial_action, initial_memory, embed)                 
                
                for t in range(starts[i]):
                    action = episodes[i][t]["action"].unsqueeze(0)
                    next_state = episodes[i][t]["next_state"].unsqueeze(0)
                    embed = self.encoder(next_state)
                    memory, _, latent, _, _ = self.rssm.observe(latent, action, memory, embed)                  
                
                memories.append(memory.squeeze(0))
                latents.append(latent.squeeze(0))
                
        memory = torch.stack(memories)
        latent = torch.stack(latents)
        
        for t in range(sequence_length):
            action = actions[:, t, :]
            reward = rewards[:, t].unsqueeze(-1)
            next_state = next_states[:, t, :]
            
            embed = self.encoder(next_state)
            memory, posterior_dist, posterior_latent, prior_dist, prior_latent = self.rssm.observe(latent, action, memory, embed)
        
            recon_loss = self.recon_loss(memory, posterior_latent, next_state)
            reward_loss = self.reward_loss(memory, posterior_latent, reward)
            dist_loss = self.dist_loss(posterior_dist, prior_dist)
        
            total_recon_loss += recon_loss
            total_reward_loss += reward_loss
            total_dist_loss += dist_loss
            
            latent = posterior_latent
            
        total_recon_loss /= sequence_length
        total_reward_loss /= sequence_length
        total_dist_loss /= sequence_length
        
        total_loss = total_recon_loss + total_reward_loss + total_dist_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), total_recon_loss.item(), total_reward_loss.item(), total_dist_loss.item()   
                           
        
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.mean(x)
        std = F.softplus(self.std(x)) + 1e-4                
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        action = torch.tanh(action)
        return action
        
                
class Critic(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=64):
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


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.world_model = WorldModel(state_dim, action_dim)
        self.actor = Actor(action_dim)
        self.critic = Critic()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def imagine_with_AC(self, memory, latent):
        action = self.actor(memory, latent)
        memory, prior_latent = self.world_model.rssm.imagine(latent, action, memory)

        pred_value = self.critic(memory, prior_latent)
        pred_reward = self.world_model.reward_model(memory, prior_latent)
        
        return pred_value, pred_reward

    def actor_loss(self, pred_reward):
        actor_loss = -pred_reward
        return actor_loss
        
    def critic_loss(self, value, pred_value):
        critic_loss = F.mse_loss(value, pred_value)
        return critic_loss
    
    def update(self, memory, latent, value):
        for param in self.world_model.parameters():
            param.requires_grad = False
        
        pred_value, pred_reward = self.imagine_with_AC(memory, latent)     
        
        actor_loss = self.actor_loss(pred_reward)
        critic_loss = self.critic_loss(value, pred_value)
        
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        for param in self.world_model.parameters():
            param.requires_grad = True
            
        return loss
        

class ReplayBuffer:
    def __init__(self, capacity=1000, batch_size=8, sequence_length=50):
        self.episodes = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
    def update_sequence_length(self):
        max_length = max(len(episode) for episode in self.episodes)
        if max_length >= 100:
            self.sequence_length = 50
        elif max_length >= 80:
            self.sequence_length = 40
        elif max_length >= 60:
            self.sequence_length = 30
        elif max_length >= 40:
            self.sequence_length = 20
        elif max_length >= 20:
            self.sequence_length = 10
        else:
            self.sequence_length = 5
        
    def add_episode(self, episode):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)            
        self.episodes.append(episode)
        
    def sample(self):
        self.update_sequence_length()
        
        episodes = []
        starts = []
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        valid_episodes = [episode for episode in self.episodes if len(episode) >= self.sequence_length]
        for _ in range(self.batch_size):
            episode = valid_episodes[np.random.randint(len(valid_episodes))]        
            start = np.random.randint(
                0,
                len(episode) - self.sequence_length + 1
            )
            sequence = episode[start:start + self.sequence_length]                     
            
            episodes.append(episode)
            starts.append(start)
            
            states.append(torch.stack([transition["state"] for transition in sequence]))
            actions.append(torch.stack([transition["action"] for transition in sequence]))
            rewards.append(torch.stack([transition["reward"] for transition in sequence]))
            next_states.append(torch.stack([transition["next_state"] for transition in sequence]))
            dones.append([transition["done"] for transition in sequence])
            
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)  
        
        return episodes, starts,states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.episodes)
        
        
if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = Agent(state_dim, action_dim)
    buffer = ReplayBuffer()
    writer = SummaryWriter(log_dir=BASE_DIR)        
    
    memory_dim, latent_dim = 256, 64
    step = 0    
    agent_dictionary = {}
        
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            num_agents = len(agent_ids)
            
            actions_for_unity = []
            for i, agent_id in enumerate(agent_ids):
                if agent_id not in agent_dictionary:
                    memory = torch.zeros(1, memory_dim)
                    latent = torch.zeros(1, latent_dim)
                    action = torch.zeros(1, action_dim)
            
                    state = states_tensor[i].unsqueeze(0)
                    embed = agent.world_model.encoder(state)
                    memory, _, latent, _, _ = agent.world_model.rssm.observe(latent, action, memory, embed)                
            
                    agent_dictionary[agent_id] = {
                        "memory": memory.squeeze(0),
                        "latent": latent.squeeze(0),
                        "action": action.squeeze(0),
                        "transitions": []
                    }
                    
                memory = agent_dictionary[agent_id]["memory"].unsqueeze(0)
                latent = agent_dictionary[agent_id]["latent"].unsqueeze(0)
                
                with torch.no_grad():
                    action = agent.actor(memory, latent)
                action = action.squeeze(0)                
                agent_dictionary[agent_id]["action"] = action
                actions_for_unity.append(action.numpy())
                
            actions_for_unity = np.array(actions_for_unity)
            env.set_actions(behavior_name, ActionTuple(continuous=actions_for_unity))
        
        env.step()
        step += 1
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                next_obs = next_decision_steps[agent_id].obs[0]
                done = False
            elif agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                next_obs = terminal_steps[agent_id].obs[0]
                done = True
            else:
                continue
            
            agent_dictionary[agent_id]["transitions"].append({
                "state": states_tensor[i],
                "action": agent_dictionary[agent_id]["action"],
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_obs).float(),
                "done": done
            })
            
            next_state = torch.from_numpy(next_obs).float().unsqueeze(0)
            next_embed = agent.world_model.encoder(next_state)
            
            action = agent_dictionary[agent_id]["action"].unsqueeze(0)
            memory = agent_dictionary[agent_id]["memory"].unsqueeze(0)
            latent = agent_dictionary[agent_id]["latent"].unsqueeze(0)
            
            memory, _, latent, _, _ = agent.world_model.rssm.observe(latent, action, memory, next_embed)                
            
            agent_dictionary[agent_id]["memory"] = memory.squeeze(0)
            agent_dictionary[agent_id]["latent"] = latent.squeeze(0)
            
            if done:
                episode = agent_dictionary[agent_id]["transitions"]
                buffer.add_episode(episode)
                del agent_dictionary[agent_id]           
        
        if len(buffer) > 0:    
            episodes, starts, states, actions, rewards, next_states, dones = buffer.sample()             
            total_loss, recon_loss, reward_loss, dist_loss = agent.world_model.update(episodes, starts, states, actions, rewards, next_states, dones)
            if step % 10 == 0:
                print(
                    step,
                    "total_loss:", total_loss,
                    "recon_loss:", recon_loss,
                    "reward_loss:", reward_loss,
                    "dist_loss:", dist_loss
                )
        

             

        
        
        
        
        
        
        
