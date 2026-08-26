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
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
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
        
    def update(self, state, latent, action, memory, reward):
        embed = self.encoder(state)
        memory, posterior_dist, posterior_latent, prior_dist, prior_latent = self.rssm.observe(latent, action, memory, embed)
        
        recon_loss = self.recon_loss(memory, posterior_latent, state)
        reward_loss = self.reward_loss(memory, posterior_latent, reward)
        dist_loss = self.dist_loss(posterior_dist, prior_dist)
        
        loss = recon_loss + reward_loss + dist_loss        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
                                   
        
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
    def __init__(self, capacity=1000, sequence_length=50):
        self.episodes = []
        self.capacity = capacity
        self.sequence_length = sequence_length
        
    def add_episode(self, episode):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)            
        self.episodes.append(episode)
        
    def sample(self):
        episode = self.episodes[np.random.randint(len(self.episodes))]        
        if len(episode) > self.sequence_length:
            start = np.random.randint(len(episode) - self.sequence_length + 1)
            episode = episode[start:start + self.sequence_length]
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for transition in episode:
            states.append(transition["state"])
            actions.append(transition["action"])
            rewards.append(transition["reward"])
            next_states.append(transition["next_state"])
            dones.append(transition["done"])
        
        return states, actions, rewards, next_states, dones
        
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
            states, actions, rewards, next_states, dones = buffer.sample()             
            memory = torch.zeros(1, memory_dim)
            latent = torch.zeros(1, latent_dim)
            
            for t in range(len(states)):
                state = states[t].unsqueeze(0)
                action = actions[t].unsqueeze(0)
                reward = rewards[t].unsqueeze(0).unsqueeze(0)

                loss = agent.world_model.update(state, latent, action, memory, reward)
        
        
        
            
        if step % 100 == 0:
            print(step)
             

        
        
        
        
        
        
        
