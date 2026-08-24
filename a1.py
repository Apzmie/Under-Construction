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
        
        return posterior_latent, prior_latent, memory, posterior_dist, prior_dist
        
    def imagine(self, latent, action, memory):
        gru_input = torch.cat([latent, action], dim=-1)
        memory = self.gru(gru_input, memory)
        
        prior_mean = self.prior_mean(memory)
        prior_std = F.softplus(self.prior_std(memory)) + 1e-4
        distribution = torch.distributions.Normal(prior_mean, prior_std)
        latent = distribution.rsample()
        
        return latent, memory

        
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
        posterior_latent, prior_latent, memory, posterior_dist, prior_dist = self.rssm.observe(latent, action, memory, embed)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
