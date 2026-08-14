import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.elu(x)
        embed = self.fc2(x)
        return embed
        
       
class RSSM(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=32, embed_dim=128):
        super().__init__()
        input_dim = latent_dim + action_dim
        
        self.W_forget_i = nn.Linear(input_dim, hidden_dim)
        self.W_forget_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_info_i = nn.Linear(input_dim, hidden_dim)
        self.W_info_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_update_i = nn.Linear(input_dim, hidden_dim)
        self.W_update_m = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_prior = nn.Linear(hidden_dim, hidden_dim)   
        self.W_prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.W_prior_std = nn.Linear(hidden_dim, latent_dim)  
        
        self.W_posterior = nn.Linear(hidden_dim + embed_dim, hidden_dim)   
        self.W_posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.W_posterior_std = nn.Linear(hidden_dim, latent_dim)  
        
    def gru(self, latent, action, memory):
        input = torch.cat([latent, action], dim=-1)
        
        input_for_forget = self.W_forget_i(input)
        memory_for_forget = self.W_forget_m(memory)
        
        forget_gate = torch.sigmoid(input_for_forget + memory_for_forget)       
        forgotten_memory = forget_gate * memory
           
        forgotten_memory_info = self.W_info_m(forgotten_memory)                    
        input_info = self.W_info_i(input)               
        summed_info = torch.tanh(forgotten_memory_info + input_info)
        
        input_for_update = self.W_update_i(input)
        memory_for_update = self.W_update_m(memory)

        update_gate = torch.sigmoid(input_for_update + memory_for_update)        
        updated_memory = (1 - update_gate) * memory + update_gate * summed_info
        return updated_memory
        
    def prior(self, memory):
        x = self.W_prior(memory)
        x = F.elu(x)        
        mean = self.W_prior_mean(x)
        std = self.W_prior_std(x)
        std = F.softplus(std) + 1e-4
        
        return mean, std
        
    def posterior(self, memory, embed):
        x = torch.cat([memory, embed], dim=-1)       
        x = self.W_posterior(x)
        x = F.elu(x)        
        mean = self.W_posterior_mean(x)
        std = self.W_posterior_std(x)
        std = F.softplus(std) + 1e-4
        
        return mean, std
        
    def sample(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def observe(self, embeds, actions, initial_memory, initial_latent):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []

        prior_means = []
        prior_stds = []

        posterior_means = []
        posterior_stds = []
        
        seq_len = embeds.shape[1]  # [B, seq_len, embed_dim]       
        for t in range(seq_len):
            embed = embeds[:, t, :]
            action = actions[:, t, :]
            
            memory = self.gru(latent, action, memory)
            prior_mean, prior_std = self.prior(memory)
            posterior_mean, posterior_std = self.posterior(memory, embed)
            
            latent = self.sample(posterior_mean, posterior_std)
            
            memories.append(memory)
            latents.append(latent)
            
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)

            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
        
        # [B, hidden_dim] × T → [B, T, hidden_dim]   
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_stds = torch.stack(posterior_stds, dim=1)
        
        return memories, latents, prior_means, prior_stds, posterior_means, posterior_stds
                
    def imagine(self, initial_memory, initial_latent, actions):
        memory = initial_memory
        latent = initial_latent
        
        memories = []
        latents = []

        prior_means = []
        prior_stds = []
        
        T = actions.shape[1]
        for t in range(T):
            action = actions[:, t]
            memory = self.gru(latent, action, memory)
            prior_mean, prior_std = self.prior(memory)
            latent = self.sample(prior_mean, prior_std)
            
            memories.append(memory)
            latents.append(latent)
            
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            
        memories = torch.stack(memories, dim=1)
        latents = torch.stack(latents, dim=1)
        
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        
        return memories, latents, prior_means, prior_stds
            

class Decoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)        
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        recon = self.fc3(x)
        return recon
        
        
class RewardModel(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, latent):
        x = torch.cat([memory, latent], dim=-1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        reward = self.fc3(x)
        return reward
          

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder(state_dim)
        self.reward_model = RewardModel()
        
    def loss(self, states, actions, rewards, initial_memory, initial_latent):
        embeds = self.encoder(states)
        memories, latents, prior_means, prior_stds, posterior_means, posterior_stds = self.rssm.observe(embeds, actions, initial_memory, initial_latent)
        
        recons = self.decoder(memories, latents)
        pred_rewards = self.reward_model(memories, latents)
        
        recon_loss = F.mse_loss(states, recons)
        reward_loss = F.mse_loss(rewards, pred_rewards)
        
        prior_dist = torch.distributions.Normal(prior_means, prior_stds)
        posterior_dist = torch.distributions.Normal(posterior_means, posterior_stds)
        dist_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        
        total_loss = recon_loss + reward_loss + dist_loss
                
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "reward_loss": reward_loss,
            "distribution_loss": dist_loss,
        }
        
        
class Agent:
    def __init__(self, state_dim, action_dim):
        self.world_model = WorldModel(state_dim, action_dim)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        
    def update_world_model(self, states, actions, rewards, initial_memory, initial_latent):
        losses = self.world_model.loss(states, actions, rewards, initial_memory, initial_latent)        
        self.world_model_optimizer.zero_grad()
        losses["total_loss"].backward()
        self.world_model_optimizer.step()
                
        return losses

        
        
        
        
        
        
        
        
                    

        
        
        
        
