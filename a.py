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
        x = self.fc2(x)
        return x
        
       
class RSSM(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256, latent_dim=32):
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
        
        
        
        
